#!/bin/sh
set -e
set -x

VERSION=0.5.0-demucs-cuda
docker tag klay-beam:demucs-cuda us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
docker tag klay-beam:demucs-cuda us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
