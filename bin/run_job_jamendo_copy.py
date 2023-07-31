import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    write_file,
)
from job_jamendo_copy.transforms import Trim


"""
Example usage:

python bin/run_job_jamendo_copy.py \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road/mp3/' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/jamendo_copy' \
    --runner Direct

python bin/run_job_jamendo_copy.py \
    --source_audio_path \
        'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/jamendo_copy' \
    --runner Direct


python bin/run_job_jamendo_copy.py \
    --region us-east1 \
    --autoscaling_algorithm THROUGHPUT_BASED \
    --runner DataflowRunner \
    --service_account_email dataset-dataflow-worker@klay-training.iam.gserviceaccount.com \
    --disk_size_gb=50 \
    --experiments=use_runner_v2 \
    --sdk_container_image=us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.2.0 \
    --sdk_location=container \
    --setup_file ./job_jamendo_copy/setup.py \
    --temp_location gs://klay-dataflow-test-000/tmp/jamendo/ \
    --project klay-training \
    --source_audio_path \
        'gs://klay-datasets/mtg_jamendo_autotagging/audios' \
    --target_audio_path \
        'gs://klay-datasets-001/mtg-jamendo-90s-crop/' \
    --job_name 'jamendo-copy-002'

    # Possible values for --source_audio_path
        'gs://klay-datasets/mtg_jamendo_autotagging/audios' \
        'gs://klay-dataflow-test-000/test-audio/abbey_road/mp3/' \
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_audio_path",
        dest="input",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.

        To get only some wav files, try:
        '/Users/alice/datasets/fma_large/005/'

        To run on the full dataset use:
        'gs://klay-datasets/mtg_jamendo_autotagging/audios/'
        """,
    )

    parser.add_argument(
        "--target_audio_path",
        dest="output",
        required=True,
        help="""
        Specify the output directory.

        For example:
        'gs://klay-dataflow-test-000/results/outputs/1/'
        """,
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Pattern to recursively find mp3s inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.mp3")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "Load Audio" >> beam.ParDo(LoadWithTorchaudio())
            | "Random Trim" >> beam.ParDo(Trim(known_args.input, known_args.output))
            | "Write Audio" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
