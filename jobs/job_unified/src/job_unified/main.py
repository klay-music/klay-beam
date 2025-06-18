#!/usr/bin/env python3
"""Main entry point for the unified audio processor."""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List

from job_unified.audio_format import AudioFormat
from job_unified.processor import ProcessorConfig, UnifiedProcessor, find_audio_files


def expand_path(path: str) -> str:
    """Expand user and environment variables in path."""
    return os.path.expandvars(os.path.expanduser(path))


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process audio files through multiple models"
    )
    
    # Input/output options
    parser.add_argument(
        "--source_audio_path",
        required=True,
        help="Path to audio file or directory containing audio files",
        type=expand_path,  # Apply path expansion when parsing
    )
    parser.add_argument(
        "--match_suffix",
        default=".mp3",
        help="Suffix to match audio files",
    )
    parser.add_argument(
        "--audio_format",
        choices=[f.value for f in AudioFormat],
        default=AudioFormat.VORBIS.value,
        help="Output audio format",
    )
    
    # Processing stages
    parser.add_argument(
        "--skip_demucs",
        action="store_true",
        help="Skip Demucs source separation",
    )
    parser.add_argument(
        "--skip_whisper",
        action="store_true",
        help="Skip Whisper transcription",
    )
    parser.add_argument(
        "--skip_byt5",
        action="store_true",
        help="Skip ByT5 processing",
    )
    parser.add_argument(
        "--skip_mtrpp",
        action="store_true",
        help="Skip MTRPP processing",
    )
    parser.add_argument(
        "--skip_klaynac",
        action="store_true",
        help="Skip KlayNAC processing",
    )
    
    # Model options
    parser.add_argument(
        "--demucs_model",
        default="hdemucs_mmi",
        help="Demucs model name",
    )
    parser.add_argument(
        "--demucs_num_stems",
        type=int,
        choices=[2, 4],
        default=2,
        help="Number of stems for Demucs (2 for vocals/instrumental, 4 for full separation)",
    )
    
    # Processing options
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of worker processes for data loading",
    )
    parser.add_argument(
        "--log_level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level",
    )
    
    return parser.parse_args()


def main() -> int:
    """Main entry point."""
    args = parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    
    # Find input files
    input_files = find_audio_files(args.source_audio_path, args.match_suffix)
    if not input_files:
        logging.error(f"No audio files found matching pattern: {args.match_suffix}")
        return 1
    
    # Create processor config
    config = ProcessorConfig(
        # Stages to run
        run_demucs=not args.skip_demucs,
        run_whisper=not args.skip_whisper,
        run_byt5=not args.skip_byt5,
        run_mtrpp=not args.skip_mtrpp,
        run_klaynac=not args.skip_klaynac,
        
        # Audio format
        audio_format=AudioFormat(args.audio_format),
        
        # Model settings
        demucs_model=args.demucs_model,
        demucs_num_stems=args.demucs_num_stems,
        
        # Processing settings
        num_workers=args.num_workers,
    )
    
    # Create and run processor
    try:
        processor = UnifiedProcessor(config)
        processor.process_files(input_files)
        return 0
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 