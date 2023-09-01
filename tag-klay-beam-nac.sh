#!/bin/sh
set -e
set -x

VERSION=0.10.0-nac
docker tag klay-beam:nac us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:nac us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
