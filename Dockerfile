# Conda + Apache Beam
#
# We need two prelimiary images for our multistage build: Apache Beam SDK and
# Conda
#
# 1. Apache Beam SDK
#
# We'll use this to get beam sdk artifacts, which are needed for for the beam
# "worker harness" (the process that connects the execution to the job). Note
# that we still need to pip install the beam SDK in the final image.
#
# Some resources with clues for how to use Beam in a custom container:
# https://cloud.google.com/dataflow/docs/guides/using-custom-containers#use_a_custom_base_image_or_multi-stage_builds
# https://github.com/apache/beam/issues/22349
#
# 2. Conda
# 
# We're managing some dependencies conda. We'll build these in a preliminary
# container and copy the resulting artifacts to our runtime container.
#
# A reference for how to do this:
# https://github.com/klay-music/analysis-service/blob/main/Dockerfile


# When launching a Beam job, we must use the same python version to launch the
# job as is used in the we use in the final image. This means that the beam
# invocation (which we'll probably run locally from a command line) must match
# this python version, which also must match the final image. 
#
# TLDR:
# py_version must match the python version specified in environment.yml
ARG py_version=3.10

# beam_version must match the version specified in pyproject.toml
ARG beam_version=2.48.0

FROM apache/beam_python${py_version}_sdk:${beam_version} as beam
ARG py_version
ARG beam_version


# Build the conda environment
FROM condaforge/mambaforge:latest as conda
ARG py_version
ARG beam_version

WORKDIR /klay/build
# TODO: use environment lock file
COPY environment.yml pyproject.toml ./
COPY  src/klay_beam/ ./src/klay_beam/

# create the conda environment in /env (intalling packages by copying)

# TODO: use environment lock file. At the moment, I'm using a hacky workaround
# to create a conda lock file in the build container. This is necessary because
# I do not yet have an appropriate environment lock file, and there is no way to
# create an environment from a yaml file while copying the packages.
RUN mamba env create --file environment.yml -p /tmp-env
RUN conda list --explicit -p /tmp-env > conda-linux-64.lock
RUN mamba create --copy -p /env --file conda-linux-64.lock && conda clean -afy

# Max's example installed gcc. I don't neet it, but I may still need to install
# other debian packages
# RUN apt-get update && apt-get install -y gcc


FROM python:${py_version}-slim as ok
ARG py_version
ARG beam_version

WORKDIR /klay/app
COPY . ./
COPY --from=conda /env /env
RUN python3 -m venv /env
RUN . /env/bin/activate
ENV PATH="/env/bin:$PATH"
RUN python3 -m pip install '.'


# Verify that the image does not have conflicting dependencies.
RUN pip check

# Copy files from official SDK image, including script/dependencies.
COPY --from=beam /opt/apache/beam /opt/apache/beam

# Set the entrypoint to Apache Beam SDK launcher.
ENTRYPOINT ["/opt/apache/beam/boot"]