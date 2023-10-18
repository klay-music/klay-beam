.PHONY: code-style type-check pre-test tests

code-style:
	flake8 src tests
	black --check src tests

type-check:
	mypy src tests --namespace-packages

pre-test: code-style type-check

tests:
	pytest tests/

docker-py310:
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py310 \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=./environment/py310.linux-64.lock" \
	.

docker-py310-torch:
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py310-torch \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=./environment/py310-torch.linux-64.lock" \
	.