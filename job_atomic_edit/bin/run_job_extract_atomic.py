import argparse
import os.path
import logging
from typing import Optional, Type, Union, List
import io
import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)
import torch
import torchaudio

from klay_beam.transforms import *

from klay_beam.torch_transforms import *
from job_atomic_edit.transforms import ExtractAtomicTriplets, VALID_EDITS, SkipCompletedMulti

"""
Job for extracting parsing stem files into (source, edit, target) triples. See job_atomic_edit/README.md for details.
"""

def torch_to_file(torch_data: torch.Tensor, sample_rate: int):
    in_memory_file_buffer = io.BytesIO()
    torchaudio.save(
        in_memory_file_buffer, torch_data, sample_rate=sample_rate, format="wav"
    )
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


edit2ix = {x: i for i, x in enumerate(VALID_EDITS)}
ix2st = {0: "src", 2: "tgt"}


DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py3.10-beam2.51.0-torch2.0"


class UngroupElements(beam.DoFn):
    def __init__(self, sample_rate: int):
        assert sample_rate in [
            24000,
            48000,
        ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.sample_rate = sample_rate

    def process(self, element):
        k, path, v = element
        for elem in list(v):
            # process your element
            for ix, el in enumerate(elem):
                if type(el) == torch.Tensor:
                    yield (
                        f"{path}.{ix2st[ix]}.{edit2ix[elem[1]]}",
                        el,
                        self.sample_rate,
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
        "--t",
        required=False,
        type=int,
        default=None,
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
    # match_pattern = os.path.join(known_args.input, f"**{known_args.audio_suffix}")
    match_patterns = [
        os.path.join(known_args.input, f"**.{stem}")
        for stem in ["source.wav", "bass.wav", "drums.wav", "other.wav", "vocals.wav"]
    ]

    new_suffixes = [f"src.{x}.wav" for x in range(len(VALID_EDITS))] + [
        f"tgt.{x}.wav" for x in range(len(VALID_EDITS))
    ]

    # instantiate atomic edit extractor here so we can use computed variables
    edit_fn = ExtractAtomicTriplets(known_args.t)
    ungroup_fn = UngroupElements(known_args.nac_input_sr)

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        audio_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | MultiMatchFiles(match_patterns)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompletedMulti(
                    old_suffix=["source.wav", "bass.wav", "drums.wav", "other.wav", "vocals.wav"],
                    new_suffix=new_suffixes,
                    check_timestamp=True,
                )
            )
            # # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        out = (
            audio_files
            | beam.Map(lambda x: (x[0].split("/")[-1].split(".")[0], x)) # TODO: test
            | "Group by track" >> beam.GroupByKey()
            | "Get Edit Triplets" >> beam.ParDo(edit_fn)
            | "Ungroup Elements" >> beam.ParDo(ungroup_fn)
        )
        # write out wav files
        (
            out
            | "CreatewavFile"
            >> beam.Map(lambda x: (x[0] + ".wav", torch_to_file(x[1], x[2])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
