# klay-beam


# Development
## Quick Start
Create `conda` environment:
```sh
conda env create -f environments/main.yml
```

## Dependencies
### conda
We use `conda` to handle python dependencies, the default `conda` environment creation is currently supported for 64-bit Linux operating system.  To create or update an environment:

```sh
conda env update -f environment.yml
```

All dependencies are either handled by `conda` or by `pip` via `conda`. The `pip` dependencies are all listed in the `pyproject.toml` file.

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
