# job_adt

Job for extracting MIDI data from drums stems using automatic drum transcription (ADT).

This job will:

1. Recursively search a path for `.drums.wav` files (`--drums_audio_path`)
1. For each audio file, if the targets already exist skip it. For example for
   `${SOURCE_PATH}/00/001.drums.wav`, if the following all exist, do not
   proceed with subsequent steps:
  - `${TARGET_PATH}/00/001.drums.mid`
1. Load the audio file, resample to 44.1kHz
1. Run ADT
1. Save results to (`--target_midi_path`) preserving the directory structure.


This job cannot be launched from OSX. As a result, there is no dedicated launch
environment. When launching, use the same environment that is used for the
docker container. `../environments/linux-64.005-adt.yml`.


```bash
# Get some deps:
apt-get update && apt-get install -y cmake curl gcc g++ libasound2-dev libjack-dev ffmpeg pkg-config unzip sox

# Get the model checkpoint
mkdir -p tmp/e-gmd_checkpoint && cd tmp/e-gmd_checkpoint && \
  curl -LO https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip && \
  unzip e-gmd_checkpoint.zip && rm e-gmd_checkpoint.zip

cd ../..

mv tmp/ job_adt/assets/ && rm -rf tmp
# `job_adt/assets/e-gmd_checkpoint` should contain the following files:
#   checkpoint
#   model.ckpt-569400.data-00000-of-00001
#   model.ckpt-569400.index
#   model.ckpt-569400.meta
```

```bash
conda env create -f ../environments/linux-64.005-adt.yml
conda activate adt
# Manually install this package
pip install -e '.[code-style, tests, type-check]'
```


```
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_adt.py \
    --source_audio_path \
        '/path/to/klay-beam/test_audio/abbey_road_48k' \
    --checkpoint_dir job_adt/assets/e-gmd_checkpoint \
    --runner Direct

# Run remote job with autoscaling
python bin/run_job_adt.py \
    --runner DataflowRunner \
    --machine_type n1-standard-2 \
    --max_num_workers=1100 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:0.2.0 \
    --sdk_location=container \
    --setup_file ./job_adt/setup.py \
    --temp_location gs://klay-dataflow-test-000/tmp/adt/ \
    --project klay-training \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/00' \
    --experiments=no_use_multiple_sdk_containers \
    --number_of_worker_harness_threads=1 \
    --job_name 'adt-000-on-jamendo-00'

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
