# Klay Beam

Helpers for running massively parallel Apache Beam jobs on audio data.

Processing large batches of audio data can be very time consuming. It is often
helpful to spin up many instances to apply transformations or extract features
from a large audio dataset. This package bundles a collection of utility methods
and examples designed to ease the process of massively parallel audio jobs with
GCP Dataflow and Apache Beam.

The core transformations include:

- File manipulations (for local filesystem and cloud storage)
  - Audio File writing and reading
  - Feature writing (and reading?)
  - file name mutations: Moving, and extension mutation
- SkipCompleted
- Audio data resampling
- Audio channel manipulation

## Example Job

The example job uses the following ingredients:
- `bin/run_job_example.py` pipeline script
- `klay-beam:0.2.0` docker container
- `environment/local-klay-beam.yml` local environment

To run the example job:
1. Talk to Charles or Max for GCP IAP permissions
2. Activate a `klay-beam` conda environment locally, (for example
   `environment/local-klay-beam.yml`)
3. Run `bin/run_job_example.py` (see example below for arguments)

```bash
# Run Locally in the klay-beam conda environment. Running locally allows you to
# use --input and --output paths on your local filesystem OR in object storage.
python bin/run_job_example.py \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/00500*.mp3' \
    --output 'test_audio/job_output/{}.wav' \
    --runner Direct
```

```bash
# Run remotely via GCP Dataflow. Should be executed in the `klay-beam` conda
# environment to ensure Beam SDK, python, and dependency parity between the
# local environment and Worker environments.
python bin/run_job_example.py \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --disk_size_gb 50 \
    --experiments=use_runner_v2 \
    --sdk_container_image us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.2.0 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/ \
    --project klay-beam-tests \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/**' \
    --output 'gs://klay-dataflow-test-000/results/outputs/17/{}.wav' \
    --job_name 'klay-audio-test-017'
```

Notes:

- When running remotely you can use the `--setup_file` option to upload a local
  package to the workers. For example `--setup_file=./klay_beam/setup.py` would
  cause `klay_beam` to be bundled as an `sdist` (when you execute
  `bin/run_job_example.py`) and installed on the worker nodes replacing any existing
  installation of `klay_beam` that may be in the docker container. Any missing
  pip dependencies specified in `pyproject.toml` will also be installed at
  runtime.
- When running on Dataflow, view the job execution details and logs at
  https://console.cloud.google.com/dataflow/jobs?project=klay-beam-tests
- options for `--autoscaling_algorithm` are `THROUGHPUT_BASED` and `NONE`

# Development
## Quick Start

Create `conda` environment. Environments labeled `local` are likely to work on
linux albeit without cuda support:

```sh
conda env create -f environments/local-klay-beam.yml
```

To create or update an environment:

```sh
conda env update -f environment/local-klay-beam.yml
```

## Docker Container

This docker will be run on all workers. When running a Beam job on GCP Dataflow
with the `--setup_file` option missing dependencies will be installed using pip.
However, to save time, large dependencies (or non-pip dependencies such as
ffmpeg 4) should be included in the docker container.

### Docker Build Steps

These steps build the Docker container and push to our GCP docker registry.

1. `cd environment/`
1. `./make-conda-lock-file.sh 002-py310` to generate a new
   `environment/conda-linux-64.002-py310.lock`. **IMPORTANT: Only run this step
   when you are changing the conda dependencies in the
   `environment/conda-linux-64.002-py310.yml` file.**
2. Run `docker build -f Dockerfile.klay-beam -t klay-beam:latest .`
3. Edit `tag-klay-beam.sh` to update the version, for example `0.2.0-rc.2`
4. Configure docker to authorize it to write to the artifact registry: `gcloud auth configure-docker us-docker.pkg.dev` (only needs to be once)
5. Run `tag-klay-beam.sh` to tag and push to GCP.

To test the container interactively:
`docker run --rm -it --entrypoint /bin/sh klay-beam:latest`

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