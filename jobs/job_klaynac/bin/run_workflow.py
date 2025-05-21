import argparse
import os
import os.path
import logging
import torch

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
    LoadWebm,
)

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
    ResampleTorchaudioTensor,
)

from job_klaynac.transforms import (
    ExtractKlayNAC,
    CropAudioGTDuration,
)


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
        "--model_type",
        required=True,
        choices=["discrete", "continuous"],
        type=str,
        help="""Wheter to extract discrete tokens or continuous embeddings""",
    )

    parser.add_argument(
        "--match_suffix",
        default=".instrumental.stem.mp3",
        help="""
        Suffix to match audio files. Default is '.instrumental.stem.mp3'
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff", ".webm", ".ogg"],
        help="""
        Which audio file extension is being used? This is also the
        audio file extension that will be replaced with the new
        feature file extension.
        """,
    )

    parser.add_argument(
        "--window_duration",
        default=205.0,
        type=float,
        help="""
        The window duration in seconds. This is used for the sliding window.
        """,
    )

    parser.add_argument(
        "--hop_duration",
        default=200.0,
        type=float,
        help="""
        The hop duration in seconds. This is used for the sliding window.
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

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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
    src_dir = known_args.input.rstrip("/") + "/"
    match_pattern = src_dir + f"**{known_args.match_suffix}"

    extract_fn = ExtractKlayNAC(
        audio_suffix=known_args.audio_suffix,
        extract_tokens=known_args.model_type == "discrete",
        window_duration=known_args.window_duration,
        hop_duration=known_args.hop_duration,
    )
    logging.info(f"Processing audio files from {match_pattern}.")

    # Instantiate load_audio_fn
    load_audio_fn = (
        LoadWebm() if known_args.audio_suffix == ".webm" else LoadWithTorchaudio()
    )

    # Run pipeline
    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        logging.info(f"GPU available: {torch.cuda.is_available()}")
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
            | "LoadAudio" >> beam.ParDo(load_audio_fn)
        )

        npy = (
            audio_files
            | "Resample: 48000 Hz"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    source_sr_hint=48_000,
                    target_sr=48_000,
                )
            )
            | f"{extract_fn}" >> beam.ParDo(extract_fn)
        )

        (
            npy
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    run()
