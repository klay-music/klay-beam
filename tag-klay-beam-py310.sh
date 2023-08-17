#!/bin/sh
set -e
set -x

VERSION=0.8.1-py310
docker tag klay-beam:py310 us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:py310 us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
