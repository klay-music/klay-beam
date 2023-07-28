# job_jamendo_copy

Initial job for copying+triming jamendo dataset


To run locally, activate a python environment that includes these pip deps.
(`../environments/osx-64-job-jamendo` should work)
- `apache_beam@2.48.0`
- this package (`job_jamendo_copy`)

```
# CD into the parent dir (one level up from this package) and run the launch script
python bin/run_job_jamendo_copy.py \
    --source_audio_path '/absolute/path/to/mp3/data/' \
    --target_audio_path '/absolute/path/to/job_output/' \
    --runner Direct
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
