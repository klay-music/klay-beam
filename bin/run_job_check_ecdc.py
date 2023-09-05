import argparse
import os.path
import logging
from typing import Union, Tuple
import io

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    ResampleAudio,
    write_file,
    numpy_to_file,
)

from job_nac.transforms import ReadEncodec

"""
Job for extracting EnCodec features:

1. Recursively search a path for `.wav` files
1. For each audio file, extract EnCodec tokens
1. Write the results to an .npy file adjacent to the source audio file

To run, activate a suitable python environment such as
``../environments/osx-64-nac.yml`.

```
# CD into the root klay_beam dir to the launch script:
python bin/run_job_check_ecdc.py \
    --runner Direct \
    --source_ecdc_path '/absolute/path/to/eced/files/'

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
    --temp_location gs://klay-dataflow-test-000/tmp/check-ecdc/ \
    --setup_file ./job_nac/setup.py \
    --sdk_container_image \
        us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.10.0-nac \
    --source_ecdc_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/00/' \
    --job_name 'check-ecdc-001-on-jamendo-00'
```

"""


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

    # Pattern to recursively find audio files inside source_audio_path
    match_pattern = os.path.join(known_args.input, f"**.ecdc")

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
            >> beam.Map(lambda x: logging.warn(f"Failed to decode .ecdc: {x}"))
            | "Count Failed" >> beam.combiners.Count.Globally()
            | "Log Failed Count"
            >> beam.Map(lambda x: logging.warn(f"Failed to decode {x} files"))
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
