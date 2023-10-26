.PHONY: code-style type-check pre-test tests

.PHONY: docker-py3.9-beam2.51.0-torch2.0
.PHONY: docker-py3.10-beam2.51.0
.PHONY: docker-py3.10-beam2.51.0-torch2.0

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

VERSION=$(shell ./get_version.sh)


#############
# python 3.9

docker-py3.9-beam2.51.0-torch2.0: BEAM_VERSION=2.51.0
docker-py3.9-beam2.51.0-torch2.0: ./environment/py3.9-torch2.0.lock
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(VERSION)-docker-py3.9-beam$(BEAM_VERSION)-torch2.0 \
	--build-arg="PY_VERSION=3.9" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

#############
# python 3.10

docker-py3.10-beam2.51.0: BEAM_VERSION=2.51.0
docker-py3.10-beam2.51.0: ./environment/py3.10.lock
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(VERSION)-py3.10-beam$(BEAM_VERSION) \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.10-beam2.51.0-torch2.0: BEAM_VERSION=2.51.0
docker-py3.10-beam2.51.0-torch2.0: ./environment/py3.10-torch2.0.lock
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(VERSION)-py3.10-beam$(BEAM_VERSION)-torch2.0 \
	--build-arg="PY_VERSION=3.10" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.
