# job_random_trim

Initial job for copying+triming an audio dataset. This job will:

1. Recursively search a path for `.mp3` files (`--source_audio_path`)
2. Load all audio files
3. For each audio file that is at least 20 seconds long, trim a random excerpt
   of 90 seconds. If the file is between 20 and 90 seconds, use the entire file.
4. Convert to `2-channel` `48kHz` audio.
5. Save that file to a different directory (`--target_audio_path`) with the
   extension `.source.wav`, preserving the directory structure.

To run, activate a python environment that includes these pip deps.
(`../environments/osx-64-job-random-trim.yml` should work)
- `apache_beam@2.48.0`
- This package (`job_random_trim`)
- Python 3.10 is required for running the job remotely in the `klay-beam:0.2.0`
  container (Python version must match in the local+remove environments).

```
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_random_trim.py \
    --source_audio_path '/absolute/path/to/mp3/data/' \
    --target_audio_path '/absolute/path/to/job_output/' \
    --runner Direct

# Run remote job with autoscaling
python bin/run_job_random_trim.py \
    --max_num_workers=32 \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --disk_size_gb=50 \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.2.0 \
    --sdk_location=container \
    --setup_file ./job_random_trim/setup.py \
    --temp_location gs://klay-beam-scratch-storage/tmp/jamendo/ \
    --project klay-training \
    --source_audio_path \
        'gs://klay-datasets/mtg_jamendo_autotagging/audios/' \
    --target_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --machine_type n1-standard-2 \
    --job_name 'jamendo-copy-008'

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \
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
