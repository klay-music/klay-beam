# job_discogs_effnet

Job for extracting DiscogsEffnet embeddings from audio data using essentia.

**NOTE:** Unlike other jobs, this job does not use the `klay_beam` package. It
uses tensorflow instead of pytorch.

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
conda env create -f ../environments/conda-linux-64.008-discogs-effnet.yml
conda activate discogs_effnet

# manually install this package
pip install -e '.[code-style, tests, type-check]'
```

```
# cd into the parent dir (one level up from this package) and run the launch script
python bin/run_job_discogs_effnet.py \
    --source_audio_path \
        '/path/to/klay-beam/test_audio/abbey_road_48k' \
    --runner Direct

# run remote job with autoscaling
python bin/run_job_discogs_effnet.py \
    --runner DataflowRunner \
    --machine_type n1-standard-2 \
    --max_num_workers=1000 \
    --region us-central1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:discogs-effnet-0.10.4 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/discogs_effnet/ \
    --project klay-training \
    --source_audio_path \
        'gs://klay-beam-tests-000/mtg-jamendo-90s-crop/00' \
    --experiments=no_use_multiple_sdk_containers \
    --number_of_worker_harness_threads=1 \
    --job_name 'discogs-effnet-001'

# If you edit the job_discogs_effnet package, but do not want to create a new docker file:
    --setup_file ./job_discogs_effnet/setup.py \

# Possible test values for --source_audio_path
    'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE

# Extra options to consider

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
