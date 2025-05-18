.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

conda-lock-all:
	for file in environment/*.yml; do \
		conda-lock --kind explicit -f $${file} -p linux-64 \
		&& mv conda-linux-64.lock $${file%.yml}.lock; \
	done
