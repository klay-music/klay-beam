.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

# Make .lock files when the corresponding .yml file changes
%.lock : %.yml
	./make-conda-lock-file.sh $<
