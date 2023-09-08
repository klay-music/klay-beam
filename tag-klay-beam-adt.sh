#!/bin/sh
set -e
set -x

VERSION=0.2.0
docker tag klay-beam-adt us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:latest
docker tag klay-beam-adt us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam-adt:$VERSION
