#!/usr/bin/env python3
"""
Standalone script to convert audio files using KlayNAC/KlayNACVAE.
This script replaces the Beam-based job_klaynac pipeline with a simple Python script.
"""

import argparse
import logging
import math
import os
import glob
from pathlib import Path
from typing import Optional, Tuple, List
import numpy as np
import torch
from torch import Tensor
import torchaudio
import time
from torch.utils.data import Dataset, DataLoader

# Import the KlayNAC models
import klay_codecs
from klay_codecs.nac import KlayNAC, KlayNACVAE


def get_device(mps_valid: bool = False) -> torch.device:
    """Get the best available device."""
    if mps_valid and hasattr(torch, "has_mps") and torch.has_mps:
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def remove_suffix(path: str, suffix: str) -> str:
    """Remove suffix from path if it exists."""
    if path.endswith(suffix):
        return path[: -len(suffix)]
    return path


def secs_to_samples(seconds: float, rate: int) -> int:
    """Convert seconds to number of samples."""
    return math.ceil(seconds * rate)


def make_frames(audio: Tensor, window_length: int, hop_length: int) -> List[Tensor]:
    """Slice audio into overlapping windows.

    Args:
        audio: Tensor of shape (D, T) to be sliced into windows.
        window_length: Length of each window in samples.
        hop_length: Hop size in samples.
    """
    _, T = audio.shape

    if T < window_length:
        return [audio]

    starts = np.arange(0, T, hop_length).tolist()
    windows = [audio[:, s : s + window_length] for s in starts]

    overlap = window_length - hop_length
    if windows[-1].shape[-1] < overlap:
        # If the last window is shorter than the overlap, drop it
        windows = windows[:-1]
    return windows


def make_fade_curves(overlap: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    """Return linear fade-in and fade-out curves."""
    fade_in = torch.linspace(0.0, 1.0, overlap, device=device)
    fade_out = torch.flip(fade_in, dims=[0])
    return fade_in, fade_out


def make_envelope(
    index: int, total: int, length: int, overlap: int, device: torch.device
) -> Tensor:
    """Create asymmetric cross-fade window for overlap-add."""
    fade_in, fade_out = make_fade_curves(overlap, device)
    w = torch.ones(length, device=device)

    if index > 0:
        # not first → ramp up
        w[:overlap] = fade_in
    if index < total - 1:
        # not last → ramp down
        w[-overlap:] = fade_out

    return w.unsqueeze(0)


def log_gpu_memory(prefix: str = ""):
    """Log current GPU memory usage with an optional prefix."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        max_allocated = torch.cuda.max_memory_allocated() / 1e9
        logging.info(
            f"{prefix}GPU Memory: "
            f"Current={allocated:.2f}GB, "
            f"Reserved={reserved:.2f}GB, "
            f"Peak={max_allocated:.2f}GB"
        )


def overlap_add(tensors: List[Tensor], hop_length: int, total_length: int) -> Tensor:
    """Linear overlap add along the time axis with asymmetric ramps."""
    if len(tensors) == 1:
        return tensors[0]

    D, window_length = tensors[0].shape
    overlap = window_length - hop_length
    device = tensors[0].device
    
    # Pre-allocate output tensor on same device as input
    out = torch.zeros(D, total_length, device=device)
    
    # Pre-compute fade curves for efficiency
    fade_in, fade_out = make_fade_curves(overlap, device)

    # Process frames in place
    for idx, frame in enumerate(tensors):
        if frame.shape[-1] != window_length:
            envelope_length = frame.shape[-1]
        else:
            envelope_length = window_length

        # Create envelope (reuse fade curves)
        w = torch.ones(envelope_length, device=device)
        if idx > 0:
            w[:overlap] = fade_in
        if idx < len(tensors) - 1:
            w[-overlap:] = fade_out
        w = w.unsqueeze(0)  # Add channel dimension

        # Apply envelope and add to output
        start = idx * hop_length
        end = start + window_length
        out[:, start:end] += frame * w

    return out


class KlayNACProcessor:
    """Processor for extracting KlayNAC embeddings from audio files."""

    def __init__(
        self,
        model_type: str = "continuous",
        audio_suffix: str = ".mp3",
        device: Optional[torch.device] = None,
        window_duration: float = 195.0,
        hop_duration: float = 190.0,
        max_duration: Optional[float] = None,
    ):
        self.model_type = model_type
        self.audio_suffix = audio_suffix
        self.device = device or get_device()
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.max_duration = max_duration
        self.extract_tokens = model_type == "discrete"

        # Initialize model
        self._setup_model()

    def _setup_model(self):
        """Initialize the KlayNAC model."""
        if self.extract_tokens:
            self.nac = KlayNAC()
            logging.info("Using KlayNAC (discrete tokens)")
        else:
            self.nac = KlayNACVAE(dummy_mode=False)
            logging.info(
                f"Using KlayNACVAE (continuous) with config: {self.nac.config}"
            )

        if hasattr(self.nac, "model"):
            self.nac.model.to(self.device)
            self.nac.model.eval()

        logging.info(f"Model loaded on device: {self.device}")

    @property
    def audio_window_length(self) -> int:
        return secs_to_samples(self.window_duration, self.nac.config.sample_rate)

    @property
    def audio_hop_length(self) -> int:
        return secs_to_samples(self.hop_duration, self.nac.config.sample_rate)

    @property
    def embed_window_length(self) -> int:
        return secs_to_samples(self.window_duration, self.nac.config.frame_rate)

    @property
    def embed_hop_length(self) -> int:
        return secs_to_samples(self.hop_duration, self.nac.config.frame_rate)

    @property
    def suffix(self) -> str:
        model_name = "klaynac" if self.extract_tokens else "klaynacvae"
        return f".{model_name}-{klay_codecs.__version__}.npy"

    def load_audio(self, audio_path: str) -> Tuple[Tensor, int]:
        """Load audio file using torchaudio."""
        audio, sr = torchaudio.load(audio_path)
        return audio, sr

    def process_audio(self, audio: Tensor, sr: int) -> Tuple[Tensor, int]:
        """Preprocess audio for model input."""
        # Resample if necessary
        if sr != self.nac.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.nac.config.sample_rate
            )
            resampler = resampler.to(self.device)
            audio = audio.to(self.device)
            audio = resampler(audio)
            sr = self.nac.config.sample_rate
        
        # Crop audio if max_duration is specified
        if self.max_duration is not None:
            max_samples = int(self.max_duration * sr)
            if audio.shape[-1] > max_samples:
                logging.info(
                    f"Cropping audio from {audio.shape[-1]/sr:.2f}s to {self.max_duration}s"
                )
                audio = audio[..., :max_samples]
        
        # Move audio to device if not already there
        if audio.device != self.device:
            audio = audio.to(self.device)

        return audio, sr

    def apply_model(self, audio: Tensor, sr: int) -> np.ndarray:
        """Apply the KlayNAC model to process audio frames."""
        log_gpu_memory("Before processing: ")
        logging.info(f"Audio shape: {audio.shape}, duration: {audio.shape[-1]/sr:.2f}s")

        # Create frames
        audio_frames = make_frames(
            audio, self.audio_window_length, self.audio_hop_length
        )
        embed_frames = []

        # Process frames
        with torch.no_grad():
            for i, audio_frame in enumerate(audio_frames):
                # Move frame to GPU just before processing
                audio_frame = audio_frame.to(self.device)
                
                if self.extract_tokens:
                    raise NotImplementedError(
                        "Token extraction is not supported. Use 'continuous' mode instead."
                    )
                else:
                    embeds, _ = self.nac.audio_to_embeds(audio_frame.unsqueeze(0))
                    # Move embeddings back to CPU immediately to free GPU memory
                    embed_frames.append(embeds[0].cpu())
                    # Clear GPU cache periodically
                    if i % 50 == 0:
                        torch.cuda.empty_cache()

                if i % 10 == 0:
                    logging.info(f"Processed frame {i+1}/{len(audio_frames)}")
                    log_gpu_memory(f"Frame {i+1}: ")

            if not embed_frames:
                raise ValueError("No frames were extracted.")

            # Move all embeddings back to GPU for overlap-add
            device_frames = [f.to(self.device) for f in embed_frames]
            log_gpu_memory("Before overlap-add: ")
            
            # Overlap-add the frames
            output_array = overlap_add(
                device_frames,
                hop_length=self.embed_hop_length,
                total_length=int((audio.shape[-1] / sr) * self.nac.config.frame_rate),
            )

            log_gpu_memory("After overlap-add: ")

            # Clear references to device tensors
            device_frames = None
            torch.cuda.empty_cache()
            
            output_array = output_array.cpu().numpy()
            
            if np.isnan(output_array).any():
                raise ValueError("NaN values detected in output array")

            log_gpu_memory("After processing: ")
        return output_array

    def get_output_path(self, input_path: str, output_dir: Optional[str] = None) -> str:
        """Generate the output file path for a given input audio file."""
        base = os.path.basename(remove_suffix(input_path, self.audio_suffix))
        filename = base + self.suffix
        if output_dir is not None:
            return os.path.join(output_dir, filename)
        else:
            return os.path.join(os.path.dirname(input_path), filename)


def find_audio_files(input_path: str, match_suffix: str) -> List[str]:
    """Find all audio files matching the pattern."""
    if os.path.isfile(input_path):
        return [input_path]

    # Use glob to find files recursively
    pattern = os.path.join(input_path, "**", f"*{match_suffix}")
    files = glob.glob(pattern, recursive=True)

    logging.info(f"Found {len(files)} files matching pattern {pattern}")
    return sorted(files)


class AudioDataset(Dataset):
    def __init__(self, audio_files: List[str], sample_rate: int, max_duration: Optional[float] = None):
        self.audio_files = audio_files
        self.sample_rate = sample_rate
        self.max_duration = max_duration

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        waveform, sr = torchaudio.load(audio_file)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        if self.max_duration is not None:
            max_samples = int(self.max_duration * self.sample_rate)
            if waveform.shape[-1] > max_samples:
                waveform = waveform[..., :max_samples]
        
        return audio_file, waveform


def main():
    parser = argparse.ArgumentParser(
        description="Convert audio files using KlayNAC/KlayNACVAE"
    )

    parser.add_argument(
        "--source_audio_path",
        required=True,
        help="Path to audio file or directory containing audio files",
    )

    parser.add_argument(
        "--output_dir", help="Output directory for embeddings (default: same as input)"
    )

    parser.add_argument(
        "--model_type",
        choices=["discrete", "continuous"],
        default="continuous",
        help="Type of model to use",
    )

    parser.add_argument(
        "--match_suffix",
        default=".instrumental.stem.mp3",
        help="Suffix to match audio files",
    )

    parser.add_argument(
        "--audio_suffix",
        choices=[".mp3", ".wav", ".aif", ".aiff", ".webm", ".ogg"],
        default=".mp3",
        help="Audio file extension",
    )

    parser.add_argument(
        "--window_duration",
        type=float,
        default=195.0,
        help="Window duration in seconds",
    )

    parser.add_argument(
        "--hop_duration", type=float, default=190.0, help="Hop duration in seconds"
    )

    parser.add_argument(
        "--max_duration",
        type=float,
        help="Maximum duration to process (crops longer audio)",
    )

    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing output files"
    )

    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Find audio files
    audio_files = find_audio_files(args.source_audio_path, args.match_suffix)

    if not audio_files:
        logging.error("No audio files found!")
        return

    # Initialize processor
    processor = KlayNACProcessor(
        model_type=args.model_type,
        audio_suffix=args.audio_suffix,
        window_duration=args.window_duration,
        hop_duration=args.hop_duration,
        max_duration=args.max_duration,
    )

    # Create dataset and dataloader
    dataset = AudioDataset(audio_files, processor.nac.config.sample_rate, args.max_duration)
    dataloader = DataLoader(
        dataset,
        batch_size=1,  # Process one file at a time
        num_workers=2,  # Use 2 worker processes for loading
        prefetch_factor=2  # Each worker will prefetch 2 samples
    )

    success_count = 0
    error_count = 0
    total_audio_duration = 0.0
    total_apply_model_time = 0.0
    program_start_time = time.time()

    for audio_file, audio in dataloader:
        audio_file = audio_file[0]  # Unbatch
        audio = audio[0]  # Unbatch
        
        try:
            output_path = processor.get_output_path(audio_file, args.output_dir)
            if os.path.exists(output_path) and not args.overwrite:
                logging.info(f"Skipping {audio_file}, output already exists: {output_path}")
                continue

            logging.info(f"Processing audio -> {output_path}")
            
            # Process with model
            audio = audio.to(processor.device)
            model_start = time.time()
            output_array = processor.apply_model(audio, processor.nac.config.sample_rate)
            model_end = time.time()
            total_apply_model_time += (model_end - model_start)
            
            # Save output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, output_array)
            logging.info(f"Saved embeddings with shape {output_array.shape} to {output_path}")
            
            duration = audio.shape[-1] / processor.nac.config.sample_rate
            total_audio_duration += duration
            success_count += 1

        except Exception as e:
            logging.error(f"Error processing {audio_file}: {str(e)}")
            error_count += 1

    total_program_time = time.time() - program_start_time
    apply_model_percentage = (total_apply_model_time / total_program_time) * 100 if total_program_time > 0 else 0

    logging.info(f"Processing complete! Success: {success_count}, Errors: {error_count}")
    logging.info(
        f"Throughput Summary:\n"
        f"Total audio duration: {total_audio_duration:.2f} seconds\n"
        f"Total processing time: {total_program_time:.2f} seconds\n"
        f"Processing speed: {total_audio_duration / total_program_time if total_program_time > 0 else 0:.2f}x realtime\n"
        f"Apply model time: {total_apply_model_time:.2f} seconds ({apply_model_percentage:.2f}%)"
    )


if __name__ == "__main__":
    main()
