import argparse
import io
import json
import numpy as np
import logging
import os
import os.path

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

from job_byt5.transforms import ExtractWhisperByT5, LoadJson


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--match_pattern",
        type=str,
        help="""
        The glob pattern to match files in the input directory.

        For example: gs://klay-datasets/mystic-fox/4**.whisper.json
        """,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""
        If set, the pipeline will overwrite existing files in the output directory.
        """,
    )

    return parser.parse_known_args(None)


def np_dict_to_npz(np_dict: dict[str, np.ndarray]):
    in_memory_file_buffer = io.BytesIO()
    np.savez(in_memory_file_buffer, **np_dict)
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


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

    # Pattern to recursively find audio files inside source_audio_path
    assert known_args.match_pattern.endswith(".json")

    extract_fn = ExtractWhisperByT5()

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        _ = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(known_args.match_pattern)
            | beam.Reshuffle()
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=extract_fn.whisper_suffix,
                    new_suffix=extract_fn.suffix,
                    check_timestamp=True,
                    overwrite=known_args.overwrite,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "Load JSON" >> beam.ParDo(LoadJson())
            | "ExtractByT5" >> beam.ParDo(extract_fn)
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], np_dict_to_npz(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
