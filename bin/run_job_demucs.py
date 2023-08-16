import argparse
import os.path
import logging

import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.options.pipeline_options import PipelineOptions
from apache_beam.options.pipeline_options import SetupOptions

from klay_beam.transforms import (
    LoadWithTorchaudio,
    ResampleAudio,
    write_file,
    numpy_to_wav,
    SkipCompleted,
)

from job_demucs.transforms import SeparateSources


"""
Example usage:
python bin/run_job_demucs.py \
    --source_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/abbey_road_48k' \
    --target_audio_path \
        '/Users/charles/projects/klay/python/klay-beam/test_audio/job_output/split' \
    --runner Direct
"""


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
        Specify the output directory.

        For example:
        'gs://klay-dataflow-test-000/results/outputs/1/'
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

    def to_wav(element):
        key, numpy_audio, sr = element
        return key, numpy_to_wav(numpy_audio, sr)

    # Pattern to recursively find mp3s inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.source.wav")

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
                    old_suffix=".source.wav",
                    new_suffix=[".bass.wav", ".drums.wav", ".other.wav", ".vocals.wav"],
                    source_dir=known_args.input,
                    target_dir=known_args.output,
                    check_timestamp=True,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
            | "Resample: 44.1k"
            >> beam.ParDo(
                ResampleAudio(
                    target_sr=44_100,
                    source_sr_hint=48_000,
                )
            )
            | "SourceSeparate"
            >> beam.ParDo(
                SeparateSources(
                    source_dir=known_args.input,
                    target_dir=known_args.output,
                    model_name="htdemucs_ft",
                )
            )
            | "Resample: 48K"
            >> beam.ParDo(
                ResampleAudio(
                    source_sr_hint=44_100,
                    target_sr=48_000,
                    output_numpy=True,
                )
            )
            | "CreateWavFile" >> beam.Map(to_wav)
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
