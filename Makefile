.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests bin
	black --check src tests bin

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/
