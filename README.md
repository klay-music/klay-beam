# klay-beam

Base repo for running Apache Beam jobs locally or on GCP via Dataflow.

Most Beam jobs require three compatible ingredients:

1. A pipeline script to define and launch the job
2. A specialized Docker image including Beam SDK and any dependencies required
   by the pipeline
3. A local python environment that also includes the Beam SDK and job dependencies

This repo includes helpers and examples for creating compatible scripts, Docker
images, and local environments, (1, 2, and 3 respectively).

To run an example job:
1. Talk to Charles or Max for GCP IAP permissions.
1. Activate a `klay-beam` conda environment locally, (for example
   `environment/osx-64-klay-beam.yml`)
1. Run `bin/run.py` (see example below for arguments).

```bash
# Run Locally in the klay-beam conda environment. Running locally allows you to
# use --input and --output paths on your local filesystem OR in object storage.
python bin/run.py \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/00500*.mp3' \
    --output 'test_audio/job_output/{}.wav' \
    --runner Direct \
    --temp_location gs://klay-dataflow-test-000/tmp/
```

```bash
# Run remotely via GCP Dataflow. Should be executed in the `klay-beam` conda
# environment to ensure Beam SDK, python, and dependency parity between the
# local environment and Worker environments.
python bin/run.py \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --disk_size_gb=50 \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.2.0 \
    --sdk_location=container \
    --temp_location gs://klay-dataflow-test-000/tmp/ \
    --project klay-beam-tests \
    --input 'gs://klay-dataflow-test-000/test-audio/fma_large/005/**' \
    --output 'gs://klay-dataflow-test-000/results/outputs/17/{}.wav' \
    --job_name 'klay-audio-test-017'
```

Notes:

- When running remotely you can use the `--setup_file` option to upload a local package to the workers. For example `--setup_file=./klay_beam/setup.py` would cause `klay_beam` to be bundled as an `sdist` (when you execute `bin/run.py`) and installed on the worker nodes replacing any existing installation of `klay_beam` that may be in the docker container. Any missing pip dependencies specified in `pyproject.toml` will also be installed at runtime.
- When running on Dataflow, view the job execution details and logs at  https://console.cloud.google.com/dataflow/jobs?project=klay-beam-tests
- options for `--autoscaling_algorithm` are `THROUGHPUT_BASED` and `NONE`

# Development
## Quick Start

Create `conda` environment. Environments labeled `osx-64` are likely to work on linux albeit without cuda support:

```sh
conda env create -f environments/osx-64-klay-beam.yml
```

To create or update an environment:

```sh
conda env update -f environment/osx-64-klay-beam.yml
```

## Docker Container

This docker will be run on all workers. When running a Beam job on GCP Dataflow,
missing dependencies will be installed using pip. However, to save time, large
or non-pip dependencies (such as ffmpeg 4) should be included in the docker
container.

### Docker Build Steps

These steps build the Docker container and push to our GCP docker registry.

1. `cd environment/`
1. `./make-conda-lock-file.sh 002-py310` to generate a new `environment/conda-linux-64.002-py310.lock`. **IMPORTANT: Only run this step when you are changing the conda dependencies in the `environment/conda-linux-64.002-py310.yml` file.**
2. Run `docker build -t klay-beam:latest .`
3. Edit `tag-klay-beam.sh` to update the version, for example `0.2.0-rc.2`
4. Run `tag-klay-beam.sh` to tag and push to GCP

To test the container interactively: `docker run --rm -it --entrypoint /bin/sh klay-beam:latest`

## Code Quality
### Testing
We use `pytest` for testing, there's no coverage target at the moment but essential functions and custom logic should definitely be tested. To run the tests:
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
