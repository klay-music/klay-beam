#!/bin/sh
set -e
set -x

VERSION=0.10.3-discogs-effnet
docker tag klay-beam:discogs-effnet us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:discogs-effnet us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
