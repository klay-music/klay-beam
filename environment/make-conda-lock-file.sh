#!/bin/sh

# During the docker build process, we create a conda environment from the
# conda-linux-64.lock file that is checked in to the git repo. If we need to
# change the conda dependencies, we need to update the lock file. This script
# will create a new lock file and copy it to the local environment directory.

set -x

NAME=cuda-001

docker build -t conda-lock-helper:$NAME -f- . <<EOF
FROM condaforge/mambaforge:latest

WORKDIR /klay/build
COPY ./conda-lock-helper.${NAME}.yml ./conda-lock-helper.${NAME}.yml
RUN mamba env create --file ./conda-lock-helper.${NAME}.yml -p /tmp-env
RUN conda list --explicit -p /tmp-env > conda-linux-64.${NAME}.lock
EOF

ID=$(docker create conda-lock-helper:$NAME)
docker cp $ID:/klay/build/conda-linux-64.${NAME}.lock ./conda-linux-64.${NAME}.lock
docker rm -v $ID
