# job_klaynac

Neural audio encoding with KlayNAC or KlayNACVAE.

# Usage
```
# Running locally / debugging
make run-local \
    source_audio_path=gs://path/to/src_dir \
    audio_suffix=.ogg \
    match_suffix=.ogg

# Running on GCP Dataflow
make run-dataflow \
    max_num_workers=10 \
    job_name=job-klaynac \
    source_audio_path=gs://path/to/src_dir \
    audio_suffix=.ogg \
    match_suffix=.ogg
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
