import argparse
import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)
import logging
import os

from job_mds.transforms import Enumerate, ProcessURI, WriteMDS


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        required=True,
        help="""
        Specify the parent directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        help="""
        Specify the parent directory where the output files will be written.
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        help="""
        Which audio file extension is being used? This is also the
        audio file extension that will be replaced by the new
        feature file extension.
        """,
    )

    parser.add_argument(
        "--min_duration",
        type=int,
        default=20,
        help="""
        Minimum duration of audio files to process.
        """,
    )

    parser.add_argument(
        "--frame_rate",
        type=int,
        default=30,
        help="""
        Frame rate of the audio files.
        """,
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()

    pipeline_opts = PipelineOptions(pipeline_args)
    pipeline_opts.view_as(SetupOptions).save_main_session = True

    if (
        pipeline_opts.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_opts.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_opts.view_as(WorkerOptions).sdk_container_image = os.environ[
            "DOCKER_IMAGE_NAME"
        ]

    # Fully-qualified glob for MatchFiles
    # src_root = known_args.src_dir.rstrip("/") + "/"
    src_root = known_args.src_dir
    match_pattern = src_root + f"**{known_args.audio_suffix}"
    logging.info(f"{match_pattern=}")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_opts) as p:
        # -------------------------------------------------------------- #
        # 1. Find candidate audio files, skip the ones already copied
        # -------------------------------------------------------------- #
        audio_files = (
            p
            | beam_io.MatchFiles(match_pattern)
            | beam.Reshuffle()
            | beam_io.ReadMatches()
            | "GetURI" >> beam.Map(lambda x: os.path.dirname(x.metadata.path))
        )

        # -------------------------------------------------------------- #
        # 2. Process the audio files and write MDS
        # -------------------------------------------------------------- #
        _ = (
            audio_files
            | "ProcessURI"
            >> beam.ParDo(ProcessURI(known_args.min_duration, known_args.frame_rate))
            | "WriteMDS" >> beam.ParDo(WriteMDS(known_args.dest_dir))
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
