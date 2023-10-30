# job_chroma

Extract chroma features from audio files.

1. Recursively search a path for `.wav` files
1. For each audio file, extract chroma features
1. Write the results as a `.npy` adjacent to the source audio file

To run, activate the conda dev+launch environment: `environment/dev.yml`.

```bash
# CD into the root klay_beam dir to the launch script:
python bin/run_job_extract_chroma.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'

# Run remote job in test sandbox GCP project. NOTE: even with only 8 parallel
# jobs per node, this was able to overload cloud storage (once, with a 503
# error) on a test dataset of 4 songs. Larger datasets should allow for more
# parallelism. Additionally, the bottleneck may be a result of all 8 connections
# from the same IP address which could be mitigated with a high-node count and
# low parallelism within each node.
python bin/run_job_extract_chroma.py \
    --project klay-beam-tests \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --machine_type n1-standard-8 \
    --region us-central1 \
    --max_num_workers 50 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --experiments use_runner_v2 \
    --sdk_location container \
    --setup_file ./setup.py \
    --temp_location gs://klay-dataflow-test-000/tmp/extract_chroma/ \
    --source_audio_path 'gs://klay-dataflow-test-000/glucose-karaoke/' \
    --job_name 'extract-chroma-test-000'

# Run remote job with autoscaling
python bin/run_job_extract_chroma.py \
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
    --temp_location gs://klay-dataflow-test-000/tmp/extract_chroma/ \
    --source_audio_path 'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --job_name 'extract-chroma-006'

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
