# job_demucs

Initial job for copying+triming an audio dataset. This job will:

1. Recursively search a path for `.source.wav` files (`--source_audio_path`)
1. For each audio file, if the targets already exist skip it. For example for
   `${SOURCE_PATH}/00/001.source.wav`, if the following all exist, do not
   proceed with subsequent steps:
  - `${TARGET_PATH}/00/001.drums.wav`
  - `${TARGET_PATH}/00/001.bass.wav`
  - `${TARGET_PATH}/00/001.vocals.wav`
  - `${TARGET_PATH}/00/001.other.wav`
1. Load the audio file, resample to 44.1kHz
1. Run source separation
1. Resample back to 48kHz
1. Save results to (`--target_audio_path`) preserving the directory structure.

To run, activate the conda dev+launch environment: `environment/dev.yml`.

```bash
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_demucs.py \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road/source.wav/' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/split' \
    --runner Direct

# Run remote job in test sandbox GCP project.
python bin/run_job_demucs.py \
    --runner DataflowRunner \
    --project klay-beam-tests \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --max_num_workers 4 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments=use_runner_v2 \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/demucs/ \
    --source_audio_path 'gs://klay-dataflow-test-000/mtg-jamendo-90s-crop/01' \
    --target_audio_path 'gs://klay-dataflow-test-000/mtg-jamendo-90s-crop/01' \
    --machine_type n2-standard-48 \
    --number_of_worker_harness_threads=24 \
    --job_name 'demucs-test-002'

# Run remote job with autoscaling
python bin/run_job_demucs.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --max_num_workers 100 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments=use_runner_v2 \
    --sdk_location=container \
    --temp_location gs://klay-beam-scratch-storage/tmp/demucs/ \
    --source_audio_path 'gs://klay-datasets-001/mtg-jamendo/' \
    --target_audio_path 'gs://klay-datasets-001/mtg-jamendo/' \
    --machine_type n2-standard-48 \
    --number_of_worker_harness_threads=24 \
    --job_name 'demucs-047'

# If you make changes to the demucs package, you can build and push a new docker
# image, or use:
    --setup_file ./setup.py \

```

# Development
## Quick Start
Install dependencies (we highly recommend creating and activating a virtual
python environment first):
```sh
pip install [-e] '.[code-style, type-check, tests]'
```

## Dependencies
### conda
We use `pip` to handle python dependencies.  To create or update an environment:

```sh
pip install [-e] '.[code-style, type-check, tests]'
```

All dependencies are listed in the `pyproject.toml` file in the 'dependencies'
section.

## Code Quality
### Testing
We use `pytest` for testing, there's no coverage target at the moment but
essential functions and custom logic should definitely be tested. To run the
tests:
```sh
make tests
```

### Code Style
We use `flake8` for linting and `black` for formatting.

```sh
make code-style
```

### Static Typing
We check static types using `mypy`.
```sh
make type-check
```
