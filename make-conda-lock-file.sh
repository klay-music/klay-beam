#!/bin/sh

# During the docker build process, we create a conda environment from the
# conda-linux-64.lock file that is checked in to the git repo. If we need to
# change the conda dependencies, we need to update the lock file. This script
# will create a new lock file and copy it to the local environment directory.

set -x

docker build -t klay-beam-conda-lock-file -f- . <<EOF
FROM condaforge/mambaforge:latest

WORKDIR /klay/build
COPY environment/docker.yml ./environment/docker.yml
RUN mamba env create --file environment/docker.yml -p /tmp-env
RUN conda list --explicit -p /tmp-env > conda-linux-64.lock
EOF

ID=$(docker create klay-beam-conda-lock-file)
docker cp $ID:/klay/build/conda-linux-64.lock ./environment/conda-linux-64.lock
docker rm -v $ID
