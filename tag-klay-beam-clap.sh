#!/bin/sh
set -e
set -x

VERSION=0.10.3-clap
docker tag klay-beam:clap us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker tag klay-beam:clap us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION

docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:latest
docker push us-docker.pkg.dev/klay-home/klay-docker/klay-beam:$VERSION
