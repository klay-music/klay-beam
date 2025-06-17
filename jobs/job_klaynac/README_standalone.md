# KlayNAC Standalone Audio Converter

This script converts audio files to KlayNAC embeddings without requiring Apache Beam. It's designed for local processing with GPU acceleration.

## Features

- **No Beam dependencies**: Pure Python script with minimal dependencies
- **GPU acceleration**: Automatically uses CUDA if available
- **Batch processing**: Process entire directories of audio files
- **Resume capability**: Skip already processed files (unless `--overwrite` is used)
- **Memory efficient**: Processes audio in sliding windows
- **Progress tracking**: Detailed logging of processing progress

## Requirements

```bash
pip install torch torchaudio numpy klay_codecs
```

## Usage

### Basic Usage

Process a single audio file:
```bash
python convert_audio_standalone.py --source_audio_path /path/to/audio.mp3
```

Process a directory of audio files:
```bash
python convert_audio_standalone.py --source_audio_path /path/to/audio/directory/
```

### Advanced Usage

```bash
python convert_audio_standalone.py \
    --source_audio_path /path/to/audio/directory/ \
    --output_dir /path/to/output/ \
    --model_type continuous \
    --match_suffix .instrumental.stem.mp3 \
    --audio_suffix .mp3 \
    --window_duration 195.0 \
    --hop_duration 190.0 \
    --max_duration 300.0 \
    --overwrite \
    --log_level INFO
```

## Arguments

- `--source_audio_path`: Path to audio file or directory (required)
- `--output_dir`: Output directory for embeddings (default: same as input)
- `--model_type`: Model type - `continuous` for KlayNACVAE or `discrete` for KlayNAC (default: continuous)
- `--match_suffix`: File suffix pattern to match (default: `.instrumental.stem.mp3`)
- `--audio_suffix`: Audio file extension to process (default: `.mp3`)
- `--window_duration`: Window duration in seconds (default: 195.0)
- `--hop_duration`: Hop duration in seconds (default: 190.0)
- `--max_duration`: Maximum audio duration to process in seconds (crops longer audio)
- `--overwrite`: Overwrite existing output files
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)

## Output

The script generates `.npy` files containing the KlayNAC embeddings. Output files are named:
- For continuous mode: `filename.klaynacvae-{version}.npy`
- For discrete mode: `filename.klaynac-{version}.npy`

## Performance

- **GPU acceleration**: The script automatically detects and uses CUDA GPUs
- **Memory management**: Audio is processed in overlapping windows to handle long files
- **Batch processing**: Multiple files are processed sequentially
- **Progress tracking**: Detailed logging shows processing progress and memory usage

## Example Output

```
2024-01-15 10:30:00,123 - INFO - Found 5 files matching pattern /audio/**/*.instrumental.stem.mp3
2024-01-15 10:30:01,456 - INFO - Using KlayNACVAE (continuous) with config: ...
2024-01-15 10:30:01,789 - INFO - Model loaded on device: cuda
2024-01-15 10:30:02,000 - INFO - Processing file 1/5: /audio/song1.instrumental.stem.mp3
2024-01-15 10:30:02,100 - INFO - Audio shape: torch.Size([2, 14745600]), duration: 307.20s
2024-01-15 10:30:05,200 - INFO - Processed frame 1/10
2024-01-15 10:30:15,300 - INFO - GPU memory: 2.3GB / 3.1GB
2024-01-15 10:30:20,400 - INFO - Saved embeddings with shape (128, 9216) to /audio/song1.klaynacvae-0.6.2.npy
```

## Differences from Beam Version

1. **Simplified execution**: No need for Beam pipeline setup
2. **Direct file I/O**: Uses standard Python file operations
3. **Immediate feedback**: Real-time progress logging
4. **GPU optimization**: Better GPU memory management for local processing
5. **Flexible input**: Can process single files or directories
6. **Resume capability**: Automatically skips already processed files 