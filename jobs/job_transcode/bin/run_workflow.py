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
import numpy as np
import os
import os.path

from klay_beam.path import remove_suffix
from klay_beam.transforms import (
    write_file,
    numpy_to_mp3,
    numpy_to_wav,
    numpy_to_file,
    SkipCompleted,
)
from klay_beam.torch_transforms import LoadWithTorchaudio

from job_transcode.transforms import LoadWebm, crop_or_skip_audio, TranscodeFn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".wav", ".mp3", ".webm"],
        help="""
        Which source audio file extension is being used?
        """,
    )

    parser.add_argument(
        "--target_audio_suffix",
        required=True,
        choices=[".npy", ".wav", ".mp3", ".ogg"],
        help="""
        Which audio file extension is being used? This is also the
        audio file extension that will will be replaced with the new
        feature file extension.
        """,
    )

    parser.add_argument(
        "--target_sample_rate",
        default=48000,
        type=int,
        help="""
        The target sample rate for the audio files. This is used to
        resample the audio files if necessary.
        """,
    )

    parser.add_argument(
        "--crop_duration",
        default=None,
        type=int,
        help="""
        The target duration for the audio files. This is used to
        crop the audio files if necessary. If the audio file is
        shorter than this duration, it will be skipped.
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
        pipeline_options.view_as(WorkerOptions).sdk_container_image = os.environ[
            "DOCKER_IMAGE_NAME"
        ]

    # Fully-qualified glob for MatchFiles
    src_dir = known_args.src_dir.rstrip("/") + "/"
    match_pattern = src_dir + f"**{known_args.audio_suffix}"

    # Define the transcode function
    transcode_fn = TranscodeFn(
        crop_duration=known_args.crop_duration,
        target_sample_rate=known_args.target_sample_rate,
        audio_suffix=known_args.audio_suffix,
        target_audio_suffix=known_args.target_audio_suffix,
    )

    # Define the skip_completed transform
    skip_completed = SkipCompleted(
        old_suffix=known_args.audio_suffix,
        new_suffix=transcode_fn.target_suffix,
        source_dir=src_dir[:-1],
        target_dir=src_dir[:-1],
        check_timestamp=False,
    )

    logging.info(f"match_pattern: {match_pattern}")

    if known_args.audio_suffix == ".webm":
        load_audio_fn = LoadWebm()
    else:
        load_audio_fn = LoadWithTorchaudio()

    logging.info(f"Load_audio_fn: {load_audio_fn}")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            | "Match Audio Files" >> beam_io.MatchFiles(match_pattern)
            | "Reshuffle Audio" >> beam.Reshuffle()
            | "SkipCompleted Audio" >> beam.ParDo(skip_completed)
            | "Read Audio Matches" >> beam_io.ReadMatches()
            | "Load Audio" >> beam.ParDo(load_audio_fn)
            | "Transcode Audio" >> beam.ParDo(transcode_fn)
            | "Persist File" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run()
