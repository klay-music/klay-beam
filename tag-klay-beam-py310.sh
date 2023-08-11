#!/bin/sh
set -e
set -x

VERSION=0.6.0-py310-rc.1
docker tag klay-beam:py310 us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:py310 us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
