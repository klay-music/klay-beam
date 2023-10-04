# job_discogs_effnet

Job for extracting DiscogsEffnet embeddings from audio data using essentia.

**NOTE:** Unlike other jobs, this job does not use the `klay_beam` package. It
uses tensorflow instead of pytorch. Magenta also has some very specific and old-
fashioned dependencies, so this requires python 3.7, which we don't want to have
to support as part of `klay_beam`. As a results, this package is a little
unconventional in its setup.

This job will:

1. Recursively search a path for `.wav` files (`--source_audio_path`)
1. For each audio file, if the targets already exist skip it. For example for
   `${SOURCE_PATH}/00/001.drums.wav`, if the following all exist, do not
   proceed with subsequent steps:
  - `${TARGET_PATH}/00/001.drums.mid`
1. Load the audio file, resample to 44.1kHz
1. Run ADT
1. Save results as `*.drums.mid` files adjacent to the `.drums.wav` files


```bash
conda env create -f ../environments/linux-64.008-discogs-effnet.yml
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


Create one Apache Beam SDK process per worker. Prevents the shared objects and
data from being replicated multiple times for each Apache Beam SDK process. See:
https://cloud.google.com/dataflow/docs/guides/troubleshoot-oom#one-sdk
    --experiments=no_use_multiple_sdk_containers
```


# Handling Tensorflow

**The Problem**

- We need [at least apache-beam 2.48](https://github.com/apache/beam/blob/master/CHANGES.md#breaking-changes-6) for `RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1` support
- Only newer versions of tensorflow (starting with 2.12.0) support protobuf versions that are acceptable to `apache_beam@2.48.0`.
- Tensorflow 2.12.0 requires python 3.8
- Magenta requires python 3.7
- apache-beam[gcp] 2.48.0 depends on protobuf<4.24.0 and >=3.20.3

**The solution**

- Regress to apache beam 2.46
- Do not use CONDA.
- This allows us to use the default python installation eliminating the need for `RUN_PYTHON_SDK_IN_DEFAULT_ENVIRONMENT=1`
- Use default python installation from Apache Beam docker image


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
