# job_clap

Beam job for extracting CLAP features:

1. Recursively search a path for `.wav` files
1. For each audio file, extract CLAP embeddings
1. Write the results to an `.npy` file adjacent to the source audio file


```bash
# Setup the environment
conda env create -f `environment/dev.yml`.
conda activate clap-dev

# on MacOS you may want to also:
# conda install nomkl

# Run locally launch script:
python bin/run_job_clap.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'

# Run remote job with autoscaling
python bin/run_job_clap.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers 100 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-beam-scratch-storage/tmp/extract-clap/ \
    --setup_file ./setup.py \
    --source_audio_path 'gs://klay-dataflow-test-001/mtg-jamendo-90s-crop/00' \
    --job_name 'extract-clap-005'
    --number_of_worker_harness_threads 1 \
    --audio_suffix .wav

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE

# Reduce the maximum number of threads that run DoFn instances. See:
# https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#reduce-threads
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
