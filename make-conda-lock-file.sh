#!/bin/sh

# During the docker build process, we create a conda environment from the
# conda-linux-64.lock file that is checked in to the git repo. If we need to
# change the conda dependencies, we need to update the lock file. This script
# will create a new lock file and copy it to the local environment directory.

# ensure that the first agrument is a filename ending in .yml
if [ -z "$1" ] || [ "${1: -4}" != ".yml" ]; then
    echo "Please provide a filename ending in .yml."
    exit 1
fi

# Extract the filename from the second argument
NAME=$(basename $1 .yml)

# Replace the .yml extension with .lock
LOCAL_LOCKFILE=$(echo $1 | sed 's/\.yml$/.lock/')
DOCKER_WORKDIR=/klay/build
DOCKER_YML=${DOCKER_WORKDIR}/conda-env.${NAME}.yml
DOCKER_LOCKFILE=${DOCKER_WORKDIR}/conda-env.${NAME}.lock

set -e
set -x

# build an image with the environment from the yaml file
docker build -t conda-lock-helper:$NAME -f- . <<EOF
FROM condaforge/mambaforge:latest

WORKDIR ${DOCKER_WORKDIR}
COPY $1 ${DOCKER_YML}
RUN mamba env create --file ${DOCKER_YML} -p /tmp-env
RUN conda list --explicit -p /tmp-env > ${DOCKER_LOCKFILE}
EOF

# copy the lock file from the image to the local directory
ID=$(docker create conda-lock-helper:${NAME})
docker cp $ID:${DOCKER_LOCKFILE} ${LOCAL_LOCKFILE}
docker rm -v $ID
