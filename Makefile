.PHONY: code-style type-check pre-test tests

code-style:
	flake8 klay_beam/src klay_beam/tests bin
	black --check klay_beam/src klay_beam/tests bin

type-check:
	mypy klay_beam/src klay_beam/tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest klay_beam/tests/
