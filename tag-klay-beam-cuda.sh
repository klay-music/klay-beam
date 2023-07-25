#!/bin/sh
set -x
VERSION=0.0.1-rc.5
docker tag klay-beam-cuda:latest us-docker.pkg.dev/klay-home/klay-docker/klay-beam-cuda:latest
docker tag klay-beam-cuda:latest us-docker.pkg.dev/klay-home/klay-docker/klay-beam-cuda:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam-cuda:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam-cuda:$VERSION
