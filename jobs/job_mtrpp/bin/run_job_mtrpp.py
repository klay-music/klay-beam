import argparse
import os
import os.path
import logging

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

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
    ResampleTorchaudioTensor,
)

from job_mtrpp.transforms import ExtractMTRPP


DEFAULT_IMAGE = os.environ.get("DOCKER_IMAGE_NAME", None)
if DEFAULT_IMAGE is None:
    raise ValueError("Please set the DOCKER_IMAGE_NAME environment variable.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_audio_path",
        dest="input",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--match_suffix",
        required=True,
        help="""
        Specify the audio file extension to search for when scanning input dir.
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff"],
        help="""
        Which audio file extension to remove when creating the output file?
        """,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""
        If set, overwrite existing files. Otherwise, skip processing for files that already have a
        corresponding output file.
        """,
    )

    parser.add_argument(
        "--max_duration",
        type=float,
        default=75.0,
        help="""
        Maximum duration of each chunk to process. Audio longer than this will be split into chunks.
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
    input_dir = known_args.input.rstrip("/") + "/"
    match_pattern = input_dir + f"**{known_args.match_suffix}"
    extract_fn = ExtractMTRPP(
        audio_suffix=known_args.audio_suffix,
        max_duration=known_args.max_duration,
    )

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        audio_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=known_args.audio_suffix,
                    new_suffix=extract_fn.suffix,
                    check_timestamp=True,
                    overwrite=known_args.overwrite,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        (
            audio_files
            | "ExtractMTRPP" >> beam.ParDo(extract_fn)
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
