import argparse
import logging
import os
import os.path
from pathlib import Path
import tensorflow.compat.v1 as tf

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from job_adt.transforms import (
    TranscribeDrumsAudio,
    LoadWithLibrosa,
    SkipCompleted,
)
from job_adt.utils import array_to_bytes, write_midi_file


"""
Example usage (writes MIDI files adjacent to the source audio files):
python bin/run_job_adt.py \
    --source_audio_path \
        '/path/to/klay-beam/test_audio/abbey_road_48k' \
    --checkpoint_dir job_adt/assets/e-gmd_checkpoint \
    --runner Direct
"""


tf.disable_v2_behavior()
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


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
        "--checkpoint_dir",
        type=Path,
        default="./job_adt/assets/e-gmd_checkpoint/",
        help="""
        Specify the checkpoint directory.

        For example:
        'assets/e-gmd_checkpoint'
        """,
    )

    args = parser.parse_known_args(None)

    # validate checkpoint dir
    assert args[0].checkpoint_dir.is_dir()
    assert len(list(args[0].checkpoint_dir.glob("*"))) == 4
    return args


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Pattern to recursively find mp3s inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.drums.wav")

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=".drums.wav",
                    new_suffix=[".drums.mid"],
                    source_dir=known_args.input,
                    target_dir=known_args.input,
                    check_timestamp=True,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithLibrosa(target_sr=44_100))
            # O&F-ADT accepts audio as a bytes object
            | "AudioTensorToBytes" >> beam.Map(array_to_bytes)
            | "TranscribeDrumsAudio"
            >> beam.ParDo(
                TranscribeDrumsAudio(
                    source_dir=known_args.input,
                    checkpoint_dir=known_args.checkpoint_dir,
                )
            )
            | "WriteMidiFile" >> beam.Map(write_midi_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()