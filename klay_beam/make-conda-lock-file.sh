#!/bin/sh

# During the docker build process, we create a conda environment from a conda
# lock file that is checked in to the git repo. If we need to change the conda
# dependencies, we need to update the lock file. This script will create a new
# lock file and copy the reulting lock file adjacent to the input .yml

# ensure that the first argument is a filename ending with .yml or .yaml
if [ -z "$1" ]; then
    echo "Please provide a filename."
    exit 1
fi

case "$1" in
    *.yml | *.yaml)
        # The filename ends with .yml or .yaml, proceed with your script
        ;;
    *)
        echo "Please provide a filename ending in .yml or .yaml"
        exit 1
        ;;
esac

NAME=$(basename $1 .yml)
LOCAL_YML_DIR=$(dirname $1)
LOCAL_YML_BASE=$(basename $1)
# Replace the .yml extension with .lock
LOCAL_LOCKFILE=$(echo $1 | sed 's/\.yml$/.lock/')

DOCKER_WORKDIR=/klay/build
DOCKER_YML=${DOCKER_WORKDIR}/conda-env.${NAME}.yml
DOCKER_LOCKFILE=${DOCKER_WORKDIR}/conda-env.${NAME}.lock

set -e
set -x

# build an image with the environment from the yaml file
docker build -t conda-lock-helper:$NAME -f- $LOCAL_YML_DIR <<EOF
FROM condaforge/mambaforge:latest

WORKDIR ${DOCKER_WORKDIR}
COPY $LOCAL_YML_BASE ${DOCKER_YML}
RUN mamba env create --file ${DOCKER_YML} -p /tmp-env
RUN conda list --explicit -p /tmp-env > ${DOCKER_LOCKFILE}
EOF

# copy the lock file from the image to the local directory
ID=$(docker create conda-lock-helper:${NAME})
docker cp $ID:${DOCKER_LOCKFILE} ${LOCAL_LOCKFILE}
docker rm -v $ID
