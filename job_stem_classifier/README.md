# job_stem_classifier

This job classifies audio files into the stem categories defined by `demucs`: `vocals`, `drums`,
`bass`, `other`. It does this based on naming conventions, this means this job should only
be run on datasets that adhere to this conventions. Currently the only dataset that does so is
the Glucose Karaoke dataset.

## Steps
1. Recursively search a path for `.wav` files
2. Extract the stem name from each audio filename
3. Write the file back as a `.<stem_name>.wav` file adjacent to the source audio file

To run, activate the conda dev+launch environment: `environment/stem_classifier.dev.yml`.

```bash
# Example invocation to run locally
python bin/run_job_stem_classifier.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --audio_suffix .wav \

python bin/run_job_stem_classifier.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --audio_suffix .wav \

# Run remote job with autoscaling
python bin/run_job_stem_classifier.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --max_num_workers 1000 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/stem-classifier/ \
    --setup_file ./setup.py \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --audio_suffix .wav \
    --machine_type n1-standard-2 \
    --number_of_worker_harness_threads 2 \
    --job_name 'stem-classifier-001'


# Possible test values for --source_audio_path
    'klay-dataflow-test-000/test-audio/abbey_road/wav' \

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
