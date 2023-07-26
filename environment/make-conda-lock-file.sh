#!/bin/sh

# During the docker build process, we create a conda environment from the
# conda-linux-64.lock file that is checked in to the git repo. If we need to
# change the conda dependencies, we need to update the lock file. This script
# will create a new lock file and copy it to the local environment directory.

NAME=$1

# ensure that the user passed in an argument, print an error and exit on failure
if [ -z "$NAME" ]; then
    echo "Please provide a name."
    echo "For example, for a file named conda-linux-64.001-cuda.yml"
    echo "\$ ./make-conda-lock-file.sh 001-cuda"
    echo ""
    echo "The name should be in the format <int>-<string> where int is a three"
    echo "digit incrementing integer and string is a hint that offers some clue"
    echo "about the contents of the image."
    exit 1
fi

set -x

# build an image with the environment from the yaml file
docker build -t conda-lock-helper:$NAME -f- . <<EOF
FROM condaforge/mambaforge:latest

WORKDIR /klay/build
COPY ./conda-linux-64.${NAME}.yml ./conda-linux-64.${NAME}.yml
RUN mamba env create --file ./conda-linux-64.${NAME}.yml -p /tmp-env
RUN conda list --explicit -p /tmp-env > conda-linux-64.${NAME}.lock
EOF

# copy the lock file from the image to the local directory
ID=$(docker create conda-lock-helper:${NAME})
docker cp $ID:/klay/build/conda-linux-64.${NAME}.lock ./conda-linux-64.${NAME}.lock
docker rm -v $ID
