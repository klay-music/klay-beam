#! /bin/bash

# Build a docker image for klay_beam using the `docker build` command. This is
# useful for building and testing docker iamges locally. It is not used in CI.
#
# You may configure this script by setting the following environment variables,
# all of which are optional:
#
# - KLAY_BEAM_VERSION=0.10.1-rc.1
# - BEAM_VERSION=2.51.0
# - PY_VERSION=3.10
# - TORCH_VERSION=1.11
# - TORCHVISION_VERSION=0.15
# - CUDA_VERSION=11.3

# Example invocations:
#
# # Python 3.9
# PY_VERSION=3.9 ./docker-build.sh
# PY_VERSION=3.9 TORCH_VERSION=2.0 ./docker-build.sh
# PY_VERSION=3.9 TORCH_VERSION=2.0 TORCHVISION_VERSION=0.15 ./docker-build.sh
#
# # PYTHON 3.10
# PY_VERSION=3.10 ./docker-build.sh
# PY_VERSION=3.10 TORCH_VERSION=2.0 ./docker-build.sh
# PY_VERSION=3.10 TORCH_VERSION=1.11 CUDA_VERSION=11.3 ./docker-build.sh
# PY_VERSION=3.10 TORCH_VERSION=2.0 INCLUDE_KLAY_DATA=True ./docker-build.sh
#
# # PYTHON 3.11
# PY_VERSION=3.11 ./docker-build.sh
# PY_VERSION=3.11 TORCH_VERSION=2.0 ./docker-build.sh

set -e

source ./docker-env-helper.sh

# check that the $LOCAL_CONDA_LOCK file exists
if [ ! -f "$LOCAL_CONDA_LOCK" ]; then
    echo "ERROR: $LOCAL_CONDA_LOCK does not exist. Please run ./make-conda-lock-file.sh first."
    exit 1
else
    echo "Using $LOCAL_CONDA_LOCK"
fi

echo "Building docker image with the following tags:"
echo "1. klay-beam:latest"
echo "2. $DOCKER_HUB_IMAGE"
echo "3. $DOCKER_GCP_IMAGE"
echo

make $LOCAL_CONDA_LOCK

set -x
docker build \
    -f Dockerfile.conda \
    -t klay-beam:latest \
    -t ${DOCKER_HUB_IMAGE} \
    -t ${DOCKER_GCP_IMAGE} \
    --build-arg="PY_VERSION=${PY_VERSION}" \
    --build-arg="LOCAL_CONDA_LOCK=${LOCAL_CONDA_LOCK}" \
    --build-arg="BEAM_VERSION=${BEAM_VERSION}" \
    .
