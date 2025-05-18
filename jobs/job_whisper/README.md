# job_whisper

Beam job for extracting lyrics from audio.

# Usage

The commands used for running the workflow are defined in the `Makefile`.
- `make run-local` is used to run the job locally. This is useful for debugging.
- `make run-dataflow` is used to run the job on the cloud using Dataflow.

For example:

```sh
# Run workflow locally
make run-local \
  match_pattern="gs://klay-datasets-test/**.vocals.ogg" \
  audio_suffix=.ogg

# Run workflow on Dataflow
make run-dataflow \
    job_name=job-whisper \
    max_num_workers=10 \
    match_pattern="gs://klay-datasets-test/**.vocals.ogg" \
    audio_suffix=.ogg
```



# Development
## Quick Start
Install dependencies (we highly recommend creating and activating a virtual
python environment first):
```sh
pip install [-e] '.[code-style, type-check, tests]'
```

## Dependencies11
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
