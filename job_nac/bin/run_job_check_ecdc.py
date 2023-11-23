import argparse
import os.path
import logging
from typing import Tuple

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)

from job_nac.transforms import ReadEncodec

"""
Job for extracting EnCodec features:

1. Recursively search a path for `.ecdc` files (EnCodec's native audio format)
1. For each file, attempt to decode the audio
1. Log the files that failed to decode

To run, activate the dev/launch environment from `environment/dev.yml`.

```
# Run local job
python bin/run_job_check_ecdc.py \
    --runner Direct \
    --source_ecdc_path '/absolute/path/to/ecdc/files/'

# Run remote job with autoscaling
python bin/run_job_check_ecdc.py \
    --runner DataflowRunner \
    --project klay-training \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --machine_type n1-standard-2 \
    --region us-central1 \
    --max_num_workers 600 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --experiments use_runner_v2 \
    --sdk_location container \
    --temp_location gs://klay-beam-scratch-storage/tmp/check-ecdc/ \
    --setup_file ./setup.py \
    --source_ecdc_path \
        'gs://klay-datasets-001/glucose-karaoke-003' \
    --job_name 'check-ecdc-karaoke-003'
```

"""

# NOTE: the dependencies versions in Docker image must match the dependencies in
# the launch/dev environment. When updating dependencies, make sure that the
# docker image you specify for remote jobs also provides the correct
# dependencies. Here's where to look for each dependency.
#
# - pyproject.toml pins:
#   - apache_beam
#   - klay_beam
# - environment/dev.yml pins:
#   - pytorch
#   - python
#
# The default docker container specified in the bin/run_job_<name>.py script
# should provide identical dependencies.
DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.12.1-py3.9-beam2.51.0-torch2.0"

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_ecdc_path",
        dest="input",
        required=True,
        help="""
        Specify a directory containing ecdc files. Can be a local path or a gs:// URL.

        To get only some wav files, try:
        '/Users/alice/datasets/fma_large/005/'

        To run on the full dataset use:
        'gs://klay-datasets/mtg_jamendo_autotagging/audios/'
        """,
    )

    return parser.parse_known_args(None)


def read_file(readable_file: beam_io.ReadableFile) -> Tuple[str, bytes]:
    path = readable_file.metadata.path
    data = readable_file.read()
    return (path, data)


def get_length_in_seconds(element):
    key, audio, sr = element

    duration = audio.shape[1] / sr
    logging.info(f"Loaded {duration:.3f} seconds of audio from {key}")

    return duration


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Set the default docker image if we're running on Dataflow
    if (
        pipeline_options.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_options.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_options.view_as(WorkerOptions).sdk_container_image = DEFAULT_IMAGE

    # Pattern to recursively find audio files inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.ecdc")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        ecdc_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "Get bytes" >> beam.Map(read_file)
        )

        audio, failed = ecdc_files | "ReadEncodec" >> beam.ParDo(
            ReadEncodec()
        ).with_outputs("failed", main="audio")

        (
            audio
            | "Get Length" >> beam.Map(get_length_in_seconds)
            | "SumLengths" >> beam.CombineGlobally(sum)
            | "Log"
            >> beam.Map(
                lambda x: logging.info(
                    f"Total length: {x:.3f} seconds ~= {x / 60:.3f} minutes"
                )
            )
        )

        (
            failed
            | "Log Failed"
            >> beam.Map(lambda x: logging.warning(f"Failed to decode .ecdc: {x}"))
            | "Count Failed" >> beam.combiners.Count.Globally()
            | "Log Failed Count"
            >> beam.Map(lambda x: logging.warning(f"Failed to decode {x} files"))
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
