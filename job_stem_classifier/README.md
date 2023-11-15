# job_stem_classifier

This job classifies audio files into the stem categories defined by `demucs`: `vocals`, `drums`,
`bass`, `other`. It does this based on naming conventions, this means this job should only
be run on datasets that adhere to this conventions. Currently the only dataset that does so is
the Glucose Karaoke dataset.

## Steps
1. Recursively search a path for `.wav` files
2. Extract the stem group from each audio filename, the stem group is one of [`bass`, `drums`, `other`, `source`, `vocals`]
3. Write the file back as a `.<stem_group>.wav` file adjacent to the source audio file
4. If a file with the same stem group already exists, we enumerate the suffix e.g. `bass`, `bass-1`, `bass-2`, etc.

To run, activate the conda dev+launch environment: `environment/stem_classifier.dev.yml`.

```bash
# Example invocation to run locally
python bin/run_job_stem_classifier.py \
    --runner Direct \
    --source_audio_path '/absolute/path/to/source.wav/files/'
    --audio_suffix .wav \
```

Currently, since we're matching on the file-level and also have inter-file dependencies because of
the way we're enumerating the stem groups, we cannot run this job in a parallel environment. When
run in a parallel environment, there is a chance that a race condition would emerge where two
files within a track are written to the new location with the exact same suffix.

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
