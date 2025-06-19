# KlayNAC Standalone Audio Converter

This script converts audio files to KlayNAC embeddings without requiring Apache Beam. It's designed for local processing with GPU acceleration and supports both local files and Google Cloud Storage.

## Features

- **No Beam dependencies**: Pure Python script with minimal dependencies
- **GPU acceleration**: Automatically uses CUDA if available
- **Batch processing**: Process entire directories of audio files
- **GCS support**: Read from and write to Google Cloud Storage
- **Resume capability**: Skip already processed files (unless `--overwrite` is used)
- **Memory efficient**: Processes audio in sliding windows
- **Progress tracking**: Detailed logging of processing progress

## Requirements

```bash
pip install -r requirements.txt
```

You'll also need to set up Google Cloud credentials if using GCS:
```bash
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/credentials.json"
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

Process files from Google Cloud Storage:
```bash
python convert_audio_standalone.py \
    --source_audio_path gs://your-bucket/dataset-name \
    --output_dataset processed-dataset-name
```

### Advanced Usage

```bash
python convert_audio_standalone.py \
    --source_audio_path gs://your-bucket/dataset-name \
    --output_dataset processed-dataset-name \
    --model_type continuous \
    --match_suffix .source.ogg \
    --audio_suffix .ogg \
    --window_duration 195.0 \
    --hop_duration 190.0 \
    --max_duration 300.0 \
    --overwrite \
    --log_level INFO \
    --num_files 100
```

## Arguments

- `--source_audio_path`: Path to audio file, directory, or GCS path (gs://bucket/dataset-name) (required)
- `--output_dataset`: Output dataset name for GCS paths (if different from input dataset)
- `--output_dir`: Output directory for embeddings (default: same as input)
- `--model_type`: Model type - `continuous` for KlayNACVAE or `discrete` for KlayNAC (default: continuous)
- `--match_suffix`: File suffix pattern to match (default: `.source.ogg`)
- `--audio_suffix`: Audio file extension to process (default: `.ogg`)
- `--window_duration`: Window duration in seconds (default: 195.0)
- `--hop_duration`: Hop duration in seconds (default: 190.0)
- `--max_duration`: Maximum audio duration to process in seconds (crops longer audio)
- `--num_files`: Maximum number of files to process (default: process all files)
- `--overwrite`: Overwrite existing output files
- `--log_level`: Logging level (default: INFO)

## GCS Path Format

The script expects GCS paths in the following format:
```
gs://bucket/dataset-name/uri0/uri0.source.ogg
gs://bucket/dataset-name/uri1/uri1.source.ogg
...
```

The output will maintain the same structure but in the specified output dataset:
```
gs://bucket/output-dataset/uri0/uri0.klaynacvae-VERSION.npy
gs://bucket/output-dataset/uri1/uri1.klaynacvae-VERSION.npy
...
```

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