# Unified Audio Processor

This job combines multiple audio processing models into a single efficient pipeline, optimized for running on a single L4 GPU instance. It processes audio files through the following stages:

1. **Demucs**: Source separation into vocals and instrumental stems
2. **Whisper**: Transcription of vocal stems
3. **ByT5**: Processing of Whisper transcriptions
4. **MTRPP**: Processing of vocal and instrumental stems
5. **KlayNAC**: Processing of original audio

## Requirements

The script requires the following dependencies:
- Python 3.9+
- PyTorch with CUDA support
- torchaudio
- demucs (installed from Facebook Research's GitHub)
- Other model dependencies (Whisper, ByT5, MTRPP, KlayNAC)

These dependencies are included in the job's `pyproject.toml`.

## Usage

```bash
python -m job_unified.main --source_audio_path PATH [options]
```

### Arguments

#### Input/Output Options
- `--source_audio_path`: Path to audio file or directory containing audio files
- `--match_suffix`: Suffix to match audio files (default: ".mp3")
- `--audio_format`: Output audio format (choices: opus, ogg, mp3, wav; default: ogg)

#### Processing Stages
- `--skip_demucs`: Skip Demucs source separation
- `--skip_whisper`: Skip Whisper transcription
- `--skip_byt5`: Skip ByT5 processing
- `--skip_mtrpp`: Skip MTRPP processing
- `--skip_klaynac`: Skip KlayNAC processing

#### Model Options
- `--demucs_model`: Demucs model name (default: "hdemucs_mmi")
- `--demucs_num_stems`: Number of stems for Demucs (2 or 4, default: 2)
  - 2: vocals and instrumental
  - 4: vocals, bass, drums, and other

#### Processing Options
- `--num_workers`: Number of worker processes for data loading (default: 2)
- `--log_level`: Logging level (DEBUG, INFO, WARNING, ERROR; default: INFO)

### Output Files

For each input file, the processor generates:

```
input.mp3 → 
├── input.vocals.{fmt}                # Demucs output
├── input.instrumental.{fmt}          # Demucs output
├── input.vocals.whisper.npz         # Whisper output
├── input.vocals.whisper_byt5.npz    # ByT5 output
├── input.vocals.mtrpp.npz           # MTRPP on vocals
├── input.instrumental.mtrpp.npz     # MTRPP on instrumental
└── input.klaynac.npz               # KlayNAC output
```

Where `{fmt}` is the chosen audio format extension (opus, ogg, mp3, or wav).

### Examples

```bash
# Basic usage - process all MP3 files in a directory
python -m job_unified.main --source_audio_path audio_dir --match_suffix .mp3

# Process a single file, skipping some stages
python -m job_unified.main --source_audio_path input.mp3 --skip_whisper --skip_byt5

# Process files with specific output format
python -m job_unified.main --source_audio_path audio_dir --audio_format opus

# Full separation with Demucs (4 stems)
python -m job_unified.main --source_audio_path audio_dir --demucs_num_stems 4
```

## Performance

The processor is optimized for running on a single L4 GPU instance:
- Processes one file at a time to manage memory
- Uses DataLoader for efficient audio loading
- Clears GPU cache between models
- Reports detailed timing statistics

## Notes

1. The processor automatically resamples input audio to 44.1kHz as required by Demucs.
2. Each stage checks for existing output files and skips processing if they exist.
3. Progress and timing information is displayed for each file and stage.
4. The script will recursively search for audio files in directories matching the specified suffix. 