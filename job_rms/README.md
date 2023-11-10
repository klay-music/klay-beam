# job_rms

Extract RMS features from audio files.

1. Recursively search a path for `.wav` files
1. For each audio file, extract RMS features
1. Write the results as a `.npy` adjacent to the source audio file

To run, activate the conda dev+launch environment: `environment/dev.yml`.

```bash
# CD into the root klay_beam dir to the launch script:
python bin/run_job_extract_rms.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'

# Run remote job with autoscaling
python bin/run_job_extract_rms.py \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-8 \
    --region us-central1 \
    --max_num_workers 550 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --experiments use_runner_v2 \
    --sdk_location container \
    --setup_file ./setup.py \
    --temp_location gs://klay-dataflow-test-000/tmp/extract_rms/ \
    --source_audio_path 'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --job_name 'extract-rms-006'

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE
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
