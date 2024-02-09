# This helper script is invoked in the following two contexts:
#
# 1. When building docker images locally (via docker-build.sh)
# 2. When building docker images in CI (via GitHub Actions)
#
# You probably do not need to run this script directly. Instead run it
# indirectly via docker-build.sh or GitHub Actions.
#
# The script accepts inputs via the following environment variables, all of
# which are optional:
#
# 1. KLAY_BEAM_VERSION
# 2. PY_VERSION
# 3. BEAM_VERSION
# 4. TORCH_VERSION
# 5. TORCHVISION_VERSION
# 6. CUDA_VERSION
#
# This script is responsible for setting default values for the first three
# (KLAY_BEAM_VERSION, PY_VERSION, and BEAM_VERSION). Any of the remaining three
# (TORCH_VERSION, TORCHVISION_VERSION, and CUDA_VERSION) may be unset or empty
# strings, which will cause them to be omitted from the docker image.
#
# For the docker build to be successful, you must have a correctly-named conda
# lock file, which can be created from a yaml file via make-conda-lock-file.sh.
# See the `./environment/ `dir for examples.

if [[ -z ${KLAY_BEAM_VERSION} ]]; then
    # Run get_version.sh and exit this script if get_version.sh fails
    KLAY_BEAM_VERSION=$(./get_version.sh)
fi

: ${PY_VERSION:=3.9}
: ${BEAM_VERSION:=2.53.0}

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
    echo "ERROR: ${LOCAL_CONDA_LOCK} does not exist" >&2
    exit 1
fi

DOCKER_HUB_IMAGE=docker.io/klaymusic/klay-beam:${DOCKER_TAG}
DOCKER_GCP_IMAGE=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:${DOCKER_TAG}

echo "The environment was successfully configured to build a klay_beam docker image."
echo ""
echo "KLAY_BEAM_VERSION=${KLAY_BEAM_VERSION}"
echo "PY_VERSION=${PY_VERSION}"
echo "BEAM_VERSION=${BEAM_VERSION}"
echo "TORCH_VERSION=${TORCH_VERSION}"
echo "TORCHVISION_VERSION=${TORCHVISION_VERSION}"
echo "CUDA_VERSION=${CUDA_VERSION}"
echo "LOCAL_CONDA_LOCK=${LOCAL_CONDA_LOCK}"
echo "DOCKER_TAG=${DOCKER_TAG}"
echo "DOCKER_HUB_IMAGE=${DOCKER_HUB_IMAGE}"
echo "DOCKER_GCP_IMAGE=${DOCKER_GCP_IMAGE}"
echo ""
