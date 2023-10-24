# job_nac

Neural audio encoding with EnCodec and Descript Audio Codec.

1. Recursively search a path for `.wav` files
1. For each audio file, extract neural audio tokens
1. Write the results as a file adjacent to the source audio file

To run, activate the conda dev+launch environment: `environment/nac.dev.yml`.

```bash
# Example invocation to run locally
python bin/run_job_extract_nac.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_name dac \
    --nac_input_sr 44100 \
    --audio_suffix .wav \

python bin/run_job_extract_nac.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --nac_name encodec \
    --nac_input_sr 48000 \
    --audio_suffix .wav \

# Run remote job with autoscaling
python bin/run_job_extract_nac.py \
    --runner DataflowRunner \
    --project klay-beam-tests \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --region us-central1 \
    --max_num_workers 1000 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract-ecdc-48k/ \
    --setup_file ./setup.py \
    --source_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --nac_name encodec \
    --nac_input_sr 48000 \
    --audio_suffix .wav \
    --machine_type n1-standard-2 \
    --number_of_worker_harness_threads 2 \
    --job_name 'extract-ecdc-002'


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

## Conda .lock file

The `environment/nac.linux-64.lock` is used for the Docker container. It is
derived from `environment/nac-linux-64.lock`. If you update the dependencies in
the `.yml` you must re-generate the lock file:

```bash
# CD into the parent directory, and run:
./make-conda-lock-file.sh job_nac/environment/nac.linux-64.yml
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
