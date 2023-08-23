#!/bin/bash
set -e

rm -rf assets; mkdir -p assets/e-gmd_checkpoint; cd assets/e-gmd_checkpoint
curl -LO https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/e-gmd_checkpoint.zip
unzip e-gmd_checkpoint.zip; rm e-gmd_checkpoint.zip
