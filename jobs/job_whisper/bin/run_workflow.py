import argparse
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
)

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
)

from job_whisper.transforms import ExtractWhisper, LoadWebm


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff", ".webm", ".ogg"],
        help="""
        Which audio file extension to search for when scanning input dir?
        """,
    )

    parser.add_argument(
        "--match_pattern",
        required=True,
        help="""
        Glob pattern to recursively find audio files.

        Example: gs://klay-datasets/mystic-fox/4**.vocals.stem.mp3
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
        "--vad_onset",
        type=float,
        default=0.7,
        help="VAD onset threshold",
    )

    parser.add_argument(
        "--vad_offset",
        type=float,
        default=0.4,
        help="VAD offset threshold",
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    assert known_args.match_pattern.endswith(known_args.audio_suffix), (
        f"Match pattern {known_args.match_pattern} must end with audio suffix "
        f"{known_args.audio_suffix}"
    )

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

    # Pattern to recursively find audio files inside source_audio_path
    logging.info(f"Processing audio files from {known_args.match_pattern}.")

    extract_fn = ExtractWhisper(vad_onset=known_args.vad_onset, vad_offset=known_args.vad_offset)

    load_audio_fn = LoadWebm() if known_args.audio_suffix == ".webm" else LoadWithTorchaudio()

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        _ = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | "Match Files" >>
            beam_io.MatchFiles(known_args.match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | f"SkipCompleted {extract_fn.suffix} files"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=known_args.audio_suffix,
                    new_suffix=extract_fn.suffix,
                    check_timestamp=False,
                    overwrite=known_args.overwrite,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(load_audio_fn)
            | "ExtractWhisper" >> beam.ParDo(extract_fn)
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
