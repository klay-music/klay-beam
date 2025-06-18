"""Main processor module for unified audio processing."""

import glob
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from job_unified.audio_format import AudioFormat, save_audio
from job_unified.utils import get_device
from job_unified.job_demucs import DemucsSeparator


def find_audio_files(input_path: str, match_suffix: str) -> List[str]:
    """Find all audio files matching the pattern."""
    if os.path.isfile(input_path):
        return [input_path]

    # Use glob to find files recursively
    pattern = os.path.join(input_path, "**", f"*{match_suffix}")
    files = glob.glob(pattern, recursive=True)

    logging.info(f"Found {len(files)} files matching pattern {pattern}")
    return sorted(files)


@dataclass
class ProcessorConfig:
    """Configuration for the unified processor."""
    # Stages to run
    run_demucs: bool = True
    run_whisper: bool = True
    run_byt5: bool = True
    run_mtrpp: bool = True
    run_klaynac: bool = True
    
    # Audio format settings
    audio_format: AudioFormat = AudioFormat.VORBIS
    
    # Model settings
    demucs_model: str = "hdemucs_mmi"
    demucs_num_stems: int = 2  # 2 for vocals/instrumental, 4 for full separation
    
    # Processing settings
    num_workers: int = 2
    target_sr: int = 44100  # Required by Demucs


class AudioDataset(Dataset):
    """Dataset for loading and preprocessing audio files."""
    
    def __init__(self, audio_paths: List[str], target_sr: int):
        self.audio_paths = audio_paths
        self.target_sr = target_sr

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx: int) -> tuple[str, torch.Tensor, int]:
        path = self.audio_paths[idx]
        audio_tensor, sr = torchaudio.load(path)
        
        # Convert to stereo if mono
        if audio_tensor.shape[0] == 1:
            audio_tensor = audio_tensor.repeat(2, 1)
        
        # Resample if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.target_sr)
            audio_tensor = resampler(audio_tensor)
            sr = self.target_sr
        
        return path, audio_tensor, sr


class UnifiedProcessor:
    """Main processor class that handles all processing stages."""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.device = get_device()
        logging.info(f"Using device: {self.device}")
        
        # Initialize models as needed
        if config.run_demucs:
            self.demucs = DemucsSeparator(
                model_name=config.demucs_model,
                num_workers=1,
                device=self.device,
            )
        else:
            self.demucs = None
            
        # TODO: Initialize other models (Whisper, ByT5, MTRPP, KlayNAC)
        # We'll add these as we implement each stage
    
    def _get_output_path(self, input_path: str, stem_name: str, ext: str) -> str:
        """Generate output path for a processed file."""
        input_path = Path(input_path)
        return str(input_path.parent / f"{input_path.stem}.{stem_name}{ext}")
    
    def _run_demucs(self, audio_path: str, audio: torch.Tensor, sr: int) -> dict[str, str]:
        """Run Demucs source separation."""
        if not self.config.run_demucs:
            return {}
            
        # Check if outputs already exist
        vocals_path = self._get_output_path(audio_path, "vocals", self.config.audio_format.value)
        instrumental_path = self._get_output_path(audio_path, "instrumental", self.config.audio_format.value)
        
        if os.path.exists(vocals_path) and os.path.exists(instrumental_path):
            logging.info(f"Skipping Demucs for {audio_path}, outputs already exist")
            return {"vocals": vocals_path, "instrumental": instrumental_path}
            
        # Process audio
        result_dict = self.demucs(audio)
        
        if self.config.demucs_num_stems == 2:
            # Mux stems for vocals/instrumental
            result_dict = {
                "vocals": result_dict["vocals"],
                "instrumental": (
                    result_dict["bass"] +
                    result_dict["drums"] +
                    result_dict["other"]
                ),
            }
            
        # Save outputs
        outputs = {}
        for stem_name, stem_audio in result_dict.items():
            output_path = self._get_output_path(
                audio_path, stem_name, self.config.audio_format.value
            )
            save_audio(
                torch.from_numpy(stem_audio),
                output_path,
                self.config.target_sr,
                self.config.audio_format,
            )
            outputs[stem_name] = output_path
            
        return outputs
    
    def process_file(self, audio_path: str, audio: torch.Tensor, sr: int) -> None:
        """Process a single audio file through all enabled stages."""
        try:
            # Move audio to device (GPU if available, CPU otherwise)
            audio = audio.to(self.device)
            
            # 1. Run Demucs
            demucs_outputs = self._run_demucs(audio_path, audio, sr)
            
            # TODO: Implement other stages
            # 2. Run Whisper on vocals
            # 3. Run ByT5 on Whisper output
            # 4. Run MTRPP on vocals and instrumental
            # 5. Run KlayNAC on original audio
            
            # Clear device cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
        except Exception as e:
            logging.error(f"Error processing {audio_path}: {str(e)}")
            raise
    
    def process_files(self, input_files: List[str]) -> None:
        """Process multiple audio files."""
        # Create dataset and dataloader
        dataset = AudioDataset(input_files, self.config.target_sr)
        dataloader = DataLoader(
            dataset,
            batch_size=1,  # Process one file at a time
            num_workers=self.config.num_workers,
            prefetch_factor=2,
        )
        
        total_files = len(input_files)
        total_time = 0
        
        # Process files
        for idx, (paths, audio_tensors, srs) in enumerate(dataloader, 1):
            path = paths[0]  # Unbatch
            audio = audio_tensors[0]  # Unbatch
            sr = srs[0].item()  # Unbatch
            
            logging.info(f"Processing file {idx}/{total_files}: {path}")
            
            t0 = time.time()
            self.process_file(path, audio, sr)
            t1 = time.time()
            elapsed_time = t1 - t0
            total_time += elapsed_time
            
            duration = audio.shape[1] / sr
            logging.info(
                f"Processed {duration:.2f}s audio in {elapsed_time:.2f}s "
                f"({duration/elapsed_time:.2f}x realtime)"
            )
        
        # Log final stats
        logging.info("\nProcessing Summary:")
        logging.info(f"Total files processed: {total_files}")
        logging.info(f"Total processing time: {total_time:.2f}s")
        logging.info(f"Average time per file: {total_time/total_files:.2f}s") 