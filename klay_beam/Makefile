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


# Extract klay_beam.__version__ via get_version.sh. Abort if the script fails.
GET_VERSION_ERROR := $(shell ./get_version.sh >/dev/null; echo $$?)
ifeq ($(GET_VERSION_ERROR),0)
    VERSION := $(shell ./get_version.sh)
else
    $(error ./get_version.sh error: $(shell ./get_version.sh))
endif


# Global variables. These can be overridden like PY_VERSION below.
BEAM_VERSION=2.51.0

#############
# python 3.9


docker-py3.9-beam2.51.0: ./environment/py3.9.lock
	$(eval PY_VERSION=3.9)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION))
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.9-beam2.51.0-torch2.0: ./environment/py3.9-torch2.0.lock
	$(eval PY_VERSION=3.9)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)-torch2.0)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.9-beam2.51.0-torch2.0-torchvision0.15: ./environment/py3.9-torch2.0-torchvision0.15.lock
	$(eval PY_VERSION=3.9)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)-torch2.0-torchvision0.15)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.9-beam2.51.0-torch1.12-cuda11.6: ./environment/py3.9-torch1.12-cuda11.6.lock
	$(eval PY_VERSION=3.9)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)-torch1.12-cuda11.6)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

#############
# python 3.10

docker-py3.10-beam2.51.0: ./environment/py3.10.lock
	$(eval PY_VERSION=3.10)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.10-beam2.51.0-torch2.0: ./environment/py3.10-torch2.0.lock
	$(eval PY_VERSION=3.10)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)-torch2.0)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.

docker-py3.10-beam2.51.0-torch1.11-cuda11.3: ./environment/py3.10-torch1.11-cuda11.3.lock
	$(eval PY_VERSION=3.10)
	$(eval TAG=$(VERSION)-py$(PY_VERSION)-beam$(BEAM_VERSION)-torch1.11-cuda11.3)
	docker build \
	-f Dockerfile.conda \
	-t klay-beam:latest \
	-t klaymusic/klay-beam:$(TAG) \
	-t us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$(TAG) \
	--build-arg="PY_VERSION=$(PY_VERSION)" \
	--build-arg="LOCAL_CONDA_LOCK=$<" \
	--build-arg="BEAM_VERSION=$(BEAM_VERSION)" \
	.
