#!/bin/sh
set -x
VERSION=0.2.0
docker tag klay-beam:latest us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:latest us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
