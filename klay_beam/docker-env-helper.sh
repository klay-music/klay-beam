: ${KLAY_BEAM_VERSION:=$(./get_version.sh)}
: ${PY_VERSION:=3.9}
: ${BEAM_VERSION:=2.51.0}

VERSIONS=""

if [[ ! -z ${TORCH_VERSION} ]]; then
    VERSIONS=${VERSIONS}-torch${TORCH_VERSION}
fi

if [[ ! -z ${TORCHVISION_VERSION} ]]; then
    VERSIONS=${VERSIONS}-torchvision${TORCHVISION_VERSION}
fi

if [[ ! -z ${CUDA_VERSION} ]]; then
    VERSIONS=${VERSIONS}-cuda${CUDA_VERSION}
fi

LOCAL_CONDA_LOCK=./environment/py${PY_VERSION}${VERSIONS}.lock
DOCKER_TAG=${KLAY_BEAM_VERSION}-py${PY_VERSION}-beam${BEAM_VERSION}${VERSIONS}


if [[ ! -f ${LOCAL_CONDA_LOCK} ]]; then
    echo "ERROR: ${LOCAL_CONDA_LOCK} does not exist"
    exit 1
fi

DOCKER_HUB_IMAGE=docker.io/klaymusic/klay-beam:${DOCKER_TAG}
DOCKER_GCP_IMAGE=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:${DOCKER_TAG}

echo "KLAY_BEAM_VERSION=${KLAY_BEAM_VERSION}"
echo "PY_VERSION=${PY_VERSION}"
echo "BEAM_VERSION=${BEAM_VERSION}"
echo "LOCAL_CONDA_LOCK=${LOCAL_CONDA_LOCK}"
echo "DOCKER_TAG=${DOCKER_TAG}"
echo "DOCKER_HUB_IMAGE=${DOCKER_HUB_IMAGE}"
echo "DOCKER_GCP_IMAGE=${DOCKER_GCP_IMAGE}"
echo ""

# echo docker build \
#     -f Dockerfile.conda \
#     -t klay-beam:latest \
#     -t ${DOCKER_HUB_IMAGE} \
#     -t ${DOCKER_GCP_IMAGE} \
#     --build-arg="PY_VERSION=${PY_VERSION}" \
#     --build-arg="LOCAL_CONDA_LOCK=${LOCAL_CONDA_LOCK}" \
#     --build-arg="BEAM_VERSION=${BEAM_VERSION}" \
#     .

# export DOCKER_HUB_IMAGE=${DOCKER_HUB_IMAGE}
# export DOCKER_GCP_IMAGE=${DOCKER_GCP_IMAGE}
