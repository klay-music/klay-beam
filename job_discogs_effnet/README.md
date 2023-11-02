# job_discogs_effnet

Job for extracting DiscogsEffnet embeddings from audio data using essentia.

This job will:

1. Recursively search a path for `.wav` files (`--source_audio_path`)
1. For each audio file, if the targets already exist skip it. For example for
   `${SOURCE_PATH}/00/001.wav`, if the following all exist, do not
   proceed with subsequent steps:
  - `${SOURCE_PATH}/00/001.discogs_effnet.npy`
1. Load the audio file, resample to 16kHz
1. extract discogs_effnet features
1. Save results as `*.discogs_effnet.npy` files adjacent to the `.wav` files


```bash
# Create the dev+launch environment
conda env create -f ../environments/dev.yml
conda activate job-discogs-effnet-dev

# To run the job locally, download the pre-trained model to models/ dir
bin/download-models.sh
```

```
# cd into the parent dir (one level up from this package) and run the launch script
python bin/run_job_discogs_effnet.py \
    --source_audio_path '/path/to/test/audio/' \
    --runner Direct
```

This job uses a custom Docker image instead of the `--setup_file` option. If you
change the of the `src` directory, you will also need to build and publish a new 
docker image:
```
make docker
make docker-push
```


```
# run remote job with autoscaling
python bin/run_job_discogs_effnet.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --region us-central1 \
    --max_num_workers 100 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-dataflow-test-000/tmp/extract-discogs-effnet/ \
    --source_audio_path 'gs://klay-dataflow-test-001/mtg-jamendo-90s-crop/00' \
    --job_name 'extract-discogs-effnet-001' \
    --machine_type n1-standard-8

# Extra options to consider

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \
    'gs://klay-dataflow-test-000/glucose-karaoke/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE

Reduce the maximum number of threads that run DoFn instances. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#reduce-threads
    --number_of_worker_harness_threads
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
