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

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
    ResampleTorchaudioTensor,
)

from job_nac.transforms import ExtractDAC, ExtractEncodec


"""
Job for extracting EnCodec features. See job_nac/README.md for details.
"""


DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-docker-py3.9-beam2.51-torch2.0"


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
        "--nac_name",
        required=True,
        choices=["dac", "encodec"],
        help="""
        Which neural audio codec should we use? Options are ['dac' or 'encodec']
        """,
    )

    parser.add_argument(
        "--nac_input_sr",
        required=True,
        type=int,
        choices=[16000, 24000, 44100, 48000],
        help="""
        Which audio sample rate should we extract from?
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff"],
        help="""
        Which audio file extension to search for when scanning input dir?
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

    # instantiate NAC extractor here so we can use computed variables
    extract_fn: Union[ExtractDAC, ExtractEncodec]
    if known_args.nac_name == "dac":
        extract_fn = ExtractDAC(known_args.nac_input_sr)
    elif known_args.nac_name == "encodec":
        extract_fn = ExtractEncodec(known_args.nac_input_sr)

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
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        npy, ecdc = (
            audio_files
            | f"Resample: {extract_fn.sample_rate}Hz"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    source_sr_hint=48_000,
                    target_sr=extract_fn.sample_rate,
                )
            )
            | "ExtractNAC" >> beam.ParDo(extract_fn).with_outputs("ecdc", main="npy")
        )

        (
            npy
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )

        (ecdc | "PersistEcdcFile" >> beam.Map(write_file))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
