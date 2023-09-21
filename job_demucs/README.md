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

To run, activate a suitable python environment such as
``../environments/osx-64-job-random-trim.yml`.

```
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_demucs.py \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road/source.wav/' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/split' \
    --runner Direct

# Run remote job with autoscaling
python bin/run_job_demucs.py \
    --runner DataflowRunner \
    --max_num_workers 1 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.10.2-demucs \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/demucs/ \
    --project klay-training \
    --setup_file job_demucs/setup.py \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo/03/' \
    --target_audio_path \
        'gs://klay-datasets-001/mtg-jamendo/03/' \
    --machine_type n2-highcpu-16 \
    --number_of_worker_harness_threads=4 \
    --disk_size_gb=50 \
    --job_name 'demucs-039-on-full-length-jamendo-03'

# If you change job_demucs
    --setup_file ./job_demucs/setup.py \

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE

# Extra options to consider

Reduce the maximum number of threads that run DoFn instances. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#reduce-threads
    --number_of_worker_harness_threads

Create one Apache Beam SDK process per worker. Prevents the shared objects and
data from being replicated multiple times for each Apache Beam SDK process. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#one-sdk
    --experiments=no_use_multiple_sdk_containers
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
