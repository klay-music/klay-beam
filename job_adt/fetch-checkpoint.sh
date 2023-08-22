#!/bin/bash
set -e

curl -LO https://storage.googleapis.com/magentadata/models/onsets_frames_transcription/maestro_checkpoint.zip
unzip maestro_checkpoint.zip; rm maestro_checkpoint.zip

rm -rf assets
mkdir assets
mv train/ assets/e-gmd_checkpoint
