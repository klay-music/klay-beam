#!/bin/sh
set -e
set -x

VERSION=0.8.0-demucs
docker tag klay-beam:demucs us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
docker tag klay-beam:demucs us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
