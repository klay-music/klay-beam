.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests bin
	black --check src tests bin

type-check:
	mypy src tests bin --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/
