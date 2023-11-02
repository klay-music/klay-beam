import argparse
import logging
import os.path
from pathlib import Path

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)

from job_stem_classifier.transforms import (
    ClassifyAudioStem,
    copy_file,
    SkipExistingTrack,
)


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

    parser.add_argument(
        "--target_audio_path",
        dest="output",
        required=True,
        help="""
        Specify the target audio file directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--stem_map_path",
        default="assets/stems_dict.json",
        type=Path,
        help="""
        Specify the path to the stem map file. This file contains a mapping from each
        of the stem groups to the stem names in the dataset. If a new dataset split is
        introduced we should make sure to update this mapping.
        """,
    )

    parser.add_argument("--audio_suffix", required=True)

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    if "glucose-karaoke" not in known_args.input:
        # ask if the user wants to continue (this would only happen when debugging)
        print("WARNING.This job is meant to be run on the glucose-karaoke bucket.")
        print("Please continue only if you are debugging in a local environment.")
        print("Continue? (y/n)")
        if input() != "y":
            return

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
        (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            # ReadMatches produces a PCollection of ReadableFile objects
            | "SkipExistingTrack"
            >> beam.ParDo(
                SkipExistingTrack(
                    known_args.input,
                    known_args.output,
                )
            )
            | beam_io.ReadMatches()
            | "ClassifyAudioStem"
            >> beam.ParDo(
                ClassifyAudioStem(
                    known_args.stem_map_path,
                    source_dir=known_args.input,
                    target_dir=known_args.output,
                )
            )
            | "CopyFile" >> beam.ParDo(copy_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
