# job_encodec

Beam job for extracting EnCodec features:

1. Recursively search a path for `.wav` files
1. For each audio file, extract EnCodec tokens
1. Write the results to an .npy file adjacent to the source audio file

To run, activate a suitable python environment such as
``../environments/osx-64-klay-beam-py310.yml`.

```bash
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_extract_encodec.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'

# See the docstring in `./bin/run_job_extract_encodec.py` for an example of
# running the job on Dataflow
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
