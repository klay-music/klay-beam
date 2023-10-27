import argparse
import os.path
import logging
from typing import Union

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)

from klay_beam.transforms import (
    SkipCompleted,
    write_file,
    numpy_to_file,
)


from job_stem_classifier.transforms import ClassifyAudioStem


"""
Job for classifying the stem name of audio files. See job_stem_classifier/README.md for details.
"""


DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py3.10"


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

    return parser.parse_known_args(None)


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
    match_pattern = os.path.join(known_args.input, f"**{known_args.audio_suffix}")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            # TODO: I don't SkipCompleted makes sense here? We're dynamically updating the
            # suffix of the file based on the filename, it's difficult to also implement this
            # logic inside the SkipCompleted method.
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "ClassifyAudioStem" >> beam.ParDo(ClassifyAudioStem())
            | "CopyFile" >> beam.ParDo(CopyFile())
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
