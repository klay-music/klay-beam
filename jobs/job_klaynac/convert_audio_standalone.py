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


def overlap_add(tensors: List[Tensor], hop_length: int, total_length: int) -> Tensor:
    """Linear overlap add along the time axis with asymmetric ramps."""
    if len(tensors) == 1:
        return tensors[0]

    D, window_length = tensors[0].shape
    overlap = window_length - hop_length
    out = torch.zeros(D, total_length, device=tensors[0].device)

    for idx, frame in enumerate(tensors):
        if frame.shape[-1] != window_length:
            envelope_length = frame.shape[-1]
        else:
            envelope_length = window_length

        envelope = make_envelope(
            idx, len(tensors), envelope_length, overlap, frame.device
        )
        start = idx * hop_length
        end = start + window_length
        out[:, start:end] += frame * envelope

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

    def process_audio(
        self,
        audio: Tensor,
        sr: int,
    ) -> Tuple[Tensor, int]:
        """Preprocess audio for model input."""
        # Resample if necessary
        if sr != self.nac.config.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=sr, new_freq=self.nac.config.sample_rate
            )
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

        # Move audio to device
        if self.device != torch.device("cpu"):
            audio = audio.to(self.device)

        return audio, sr

    def apply_model(self, audio: Tensor, sr: int) -> np.ndarray:
        """Apply the KlayNAC model to process audio frames."""
        logging.info(f"Audio shape: {audio.shape}, duration: {audio.shape[-1]/sr:.2f}s")

        # Create frames
        audio_frames = make_frames(
            audio, self.audio_window_length, self.audio_hop_length
        )
        embed_frames = []

        # Process frames
        with torch.no_grad():
            for i, audio_frame in enumerate(audio_frames):
                if self.extract_tokens:
                    raise NotImplementedError(
                        "Token extraction is not supported. Use 'continuous' mode instead."
                    )
                else:
                    embeds, _ = self.nac.audio_to_embeds(audio_frame.unsqueeze(0))

                embed_frames.append(embeds[0].detach().cpu())

                if i % 10 == 0:
                    logging.info(f"Processed frame {i+1}/{len(audio_frames)}")

            if not embed_frames:
                raise ValueError("No frames were extracted.")

            # Overlap-add the frames
            output_array = overlap_add(
                embed_frames,
                hop_length=self.embed_hop_length,
                total_length=int((audio.shape[-1] / sr) * self.nac.config.frame_rate),
            )

            if self.device != torch.device("cpu"):
                logging.info(
                    f"GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                    f" / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                )

            output_array = output_array.detach().cpu().numpy()

            if np.isnan(output_array).any():
                raise ValueError("NaN values detected in output array")

            return output_array


def find_audio_files(input_path: str, match_suffix: str) -> List[str]:
    """Find all audio files matching the pattern."""
    if os.path.isfile(input_path):
        return [input_path]

    # Use glob to find files recursively
    pattern = os.path.join(input_path, "**", f"*{match_suffix}")
    files = glob.glob(pattern, recursive=True)

    logging.info(f"Found {len(files)} files matching pattern {pattern}")
    return sorted(files)


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

    # Process files
    success_count = 0
    error_count = 0
    total_audio_duration = 0.0
    start_time = time.time()

    # Timing statistics
    time_loading = 0.0
    time_processing = 0.0
    time_model = 0.0
    time_saving = 0.0

    for i, audio_file in enumerate(audio_files):
        logging.info(f"Processing file {i+1}/{len(audio_files)}: {audio_file}")

        # Generate output path
        if args.output_dir:
            output_path = os.path.join(
                args.output_dir,
                os.path.basename(remove_suffix(audio_file, args.audio_suffix))
                + processor.suffix,
            )
        else:
            output_path = (
                remove_suffix(audio_file, args.audio_suffix) + processor.suffix
            )

        if os.path.exists(output_path) and not args.overwrite:
            logging.info(f"Skipping {audio_file}, output already exists: {output_path}")
            continue

        try:
            # Load audio
            t0 = time.time()
            audio, sr = processor.load_audio(audio_file)
            t1 = time.time()
            time_loading += t1 - t0

            # Track audio duration
            duration = audio.shape[-1] / sr
            if processor.max_duration is not None:
                duration = min(duration, processor.max_duration)
            total_audio_duration += duration

            # Preprocess audio
            t0 = time.time()
            audio, sr = processor.process_audio(audio, sr)
            t1 = time.time()
            time_processing += t1 - t0

            logging.info(f"Processing audio -> {output_path}")

            # Apply model
            t0 = time.time()
            output_array = processor.apply_model(audio, sr)
            t1 = time.time()
            time_model += t1 - t0

            # Save output
            t0 = time.time()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            np.save(output_path, output_array)
            logging.info(
                f"Saved embeddings with shape {output_array.shape} to {output_path}"
            )
            t1 = time.time()
            time_saving += t1 - t0

            success_count += 1
        except Exception as e:
            logging.error(f"Error processing {audio_file}: {e}")
            error_count += 1

    total_time = time.time() - start_time
    throughput = total_audio_duration / total_time if total_time > 0 else 0

    # Calculate percentages
    pct_loading = 100 * time_loading / total_time if total_time > 0 else 0
    pct_processing = 100 * time_processing / total_time if total_time > 0 else 0
    pct_model = 100 * time_model / total_time if total_time > 0 else 0
    pct_saving = 100 * time_saving / total_time if total_time > 0 else 0
    pct_other = 100 - (pct_loading + pct_processing + pct_model + pct_saving)

    logging.info(
        f"Processing complete! Success: {success_count}, Errors: {error_count}"
    )
    logging.info(
        f"Throughput Summary:\n"
        f"Total audio duration: {total_audio_duration:.2f} seconds\n"
        f"Total processing time: {total_time:.2f} seconds\n"
        f"Processing speed: {throughput:.2f}x realtime\n"
        f"\n"
        f"Time breakdown:\n"
        f"Loading audio:     {time_loading:.2f}s ({pct_loading:.1f}%)\n"
        f"Processing audio:  {time_processing:.2f}s ({pct_processing:.1f}%)\n"
        f"Applying model:    {time_model:.2f}s ({pct_model:.1f}%)\n"
        f"Saving output:     {time_saving:.2f}s ({pct_saving:.1f}%)\n"
        f"Other:            {total_time - (time_loading + time_processing + time_model + time_saving):.2f}s ({pct_other:.1f}%)"
    )


if __name__ == "__main__":
    main()
