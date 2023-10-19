.PHONY: code-style type-check pre-test tests docker-py310 docker-py310-torch

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

# Make .lock files when the corresponding .yml file changes
%.lock : %.yml ; ./make-conda-lock-file.sh $<

docker-py310: ./environment/py310.linux-64.lock
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py310 \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	.

docker-py310-torch: ./environment/py310-torch.linux-64.lock
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py310-torch \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	.