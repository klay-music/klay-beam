#! /bin/bash

# Example inputs
# BEAM_VERSION=2.51.0
# PY_VERSION=3.10
# TORCH_VERSION=1.11
# TORCHVISION_VERSION=0.15
# CUDA_VERSION=11.3

# Extract klay_beam.__version__ via get_version.sh. Abort if the script fails.
set -e
KLAY_BEAM_VERSION=$(./get_version.sh)

source ./docker-env-helper.sh

# check that the $LOCAL_CONDA_LOCK file exists
if [ ! -f "$LOCAL_CONDA_LOCK" ]; then
    echo "ERROR: $LOCAL_CONDA_LOCK does not exist. Please run ./make-conda-lock-file.sh first."
    exit 1
else
    echo "Using $LOCAL_CONDA_LOCK"
fi

echo "Building docker image with the following tags:"
echo klay-beam:latest
echo $DOCKER_HUB_IMAGE
echo $DOCKER_GCP_IMAGE
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
