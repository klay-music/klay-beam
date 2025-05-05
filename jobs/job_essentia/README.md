# job_essentia

Job for extracting `essentia` classifier features from an audio signal.

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

1. Update the job package version in `src/job_essentia/__init__.py`, for example to
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
