# job_essentia_features

Job for extracting `essentia` classifier features from an audio signal.

This job will:

1. Recursively search a `--source_audio_path` for files ending with
   `--match_suffix`.
2. Load the audio file, resample to 16kHz
3. Run classification(s) and return tuple of filepath and feature array
4. Save results as `*.<feature_name>.npy` files adjacent to the audio files

All available features are defined in the `EssentiaFeatures` enum class in `transforms.py`.

## Development

```bash
# Create the development environment
conda env create -f ../environments/dev.yml
conda activate essentia-features-dev

# Download the models
./bin/download-models.sh
```

### Launch
```
python bin/run_job_essentia_features.py \
    --runner Direct
    --source_audio_path '/path/to/test/audio/' \
    --match_suffix .instrumental.stem.mp3 \
    --audio_suffix .mp3 \
    --features voice_instrumental,mood_happy,mtt
```

```
# Run remote job on test dataset
python bin/run_job_essentia_features.py \
    --runner DataflowRunner \
    --project klay-beam-tests \
    --service_account_email dataset-dataflow-worker@klay-beam-tests.iam.gserviceaccount.com \
    --region us-central1 \
    --max_num_workers 100 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-beam-scratch-storage/tmp/extract-vocal-classifier/ \
    --source_audio_path 'gs://klay-dataflow-test-001/pretraining' \
    --job_name 'job-essentia-features' \
    --machine_type n1-standard-8 \
    --setup_file ./setup.py \
    --match_suffix .instrumental.stem.mp3 \
    --audio_suffix .mp3 \
    --features voice_instrumental,mood_happy,mtt

# Options for --autoscaling-algorithm
    THROUGHPUT_BASED, NONE
```

# Docker Images

This job uses a custom docker image defined in [./Dockerfile](Dockerfile). When
you make a change to this job package you have two options for how to ensure
that the updated package will be used when running the job via Dataflow:

1. Use the `--setup_file ./setup.py` flag when laynching the job. This will
   bundle the job package as Python Source Distribution ("sdist"), upload the
   bundled package to cloud storage, and then install the sdist inside of
   existing docker container at runtime, replacing the older job package that is
   bundled in the job's default Docker container. This will also install missing
   `pyproject.toml` dependencies in containers at runtime.
2. Alternatively, you can create a new Docker container, upload it to our
   private Docker repository on GCP, and then update the job launch script to
   specify the new docker container.

## Creating, publishing, and using Docker images

Here's how to create, push, and use an updated custom docker container:

1. Update the job package version in `src/job_essentia_features/__init__.py`, for example to
   `__version__ = "0.1.1"`
2. Update the tag version in the `DOCKER_IMAGE_NAME` environment variable defined in `.env`,
   for example `DOCKER_IMAGE_NAME=us-docker.pkg.dev/klay-home/klay-docker/klay-beam-t5:0.1.1`.
   We use `direnv` to activate the variable defined in `.env`, to make sure `direnv` is correctly
   configured look at the section below this.
3. Build the image locally with:
   ```
   make docker
   ```
4. Push the local image to to our private Docker repository:
   ```
   make docker-push
   ```

Note that in order push to our private docker repository you must:
1. Have permission to write the GCP Artifact Registry, for example via the "Artifact Registry Writer" role in the `klay-home` GCP project.
2. Setup docker to [authenticate with GCP](https://cloud.google.com/artifact-registry/docs/docker/authentication).
3. Be aware of how [docker build interacts with your machines architecture](https://stackoverflow.com/q/74942945/702912). If you build images in an ARM machine such as a MacBook with an M1 processor, you may want to add the `--platform linux/amd64` flag to `docker build`.


# Development
## Quick Start
Install dependencies (we highly recommend creating and activating a virtual
python environment first):
```sh
pip install [-e] '.[code-style, type-check, tests]'
```

# Development
## Quick Start
Install dependencies (we highly recommend creating and activating a virtual `python` environment first):
```sh
pip install [-e] '.[code-style, type-check, tests]'
```

## Dependencies
### conda
We use `pip` to handle python dependencies.  To create or update an environment:

```sh
pip install [-e] '.[code-style, type-check, tests]'
```

All dependencies are listed in the `pyproject.toml` file in the 'dependencies' section.

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
