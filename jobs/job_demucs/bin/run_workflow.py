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
    write_file,
)
from klay_beam.torch_transforms import ResampleTorchaudioTensor

from job_demucs.transforms import (
    FilterVocalAudio,
    SeparateSources,
    CropAudioGTDuration,
    LoadWithTorchaudioDebug,
    LoadWebm,
    SkipCompleted,
    numpy_to_vorbis,
)


TARGET_AUDIO_SUFFIX = ".ogg"


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--audio_suffix",
        required=True,
        default=".source.stem.mp3",
        help="""
        Which audio suffix should be replaced on the output files?
        """,
    )

    parser.add_argument(
        "--match_pattern",
        required=True,
        help="""
        Specify the match pattern to use when searching for audio files.

        For example: "gs://klay-datasets-pretraining/mystic-fox-full/4**.source.stem.mp3"
        """,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="""
        Should we overwrite existing source separated stems?
        """,
    )

    parser.add_argument(
        "--only_if_vocals",
        action="store_true",
        help="""
        Should we only separate the sources if the audio contains vocals?
        """,
    )

    parser.add_argument(
        "--p_threshold",
        type=float,
        default=0.9,
        help="Threshold for the probability of a frame containing vocals",
    )

    parser.add_argument(
        "--n_threshold",
        type=float,
        default=0.1,
        help="Threshold for the number of frames containing vocals",
    )

    parser.add_argument(
        "--max_duration",
        type=float,
        default=360.0,
        help="""
        To avoid out-of-memory errors you can specify a max_duration for audio
        files. We skip any files greater than this duration
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

    # Set the default docker image if we"re running on Dataflow
    if (
        pipeline_options.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_options.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_options.view_as(WorkerOptions).sdk_container_image = os.environ[
            "DOCKER_IMAGE_NAME"
        ]

    def to_ogg(element):
        key, numpy_audio, sr = element
        return key, numpy_to_vorbis(numpy_audio, sr)

    input_dir = known_args.match_pattern.rsplit("/", 1)[0]

    skip_completed = SkipCompleted(
        old_suffix=f".source{known_args.audio_suffix}",
        new_suffix=f".vocals{TARGET_AUDIO_SUFFIX}",
        source_dir=input_dir,
        target_dir=input_dir,
        check_timestamp=False,
        overwrite=known_args.overwrite,
    )

    if known_args.audio_suffix == ".webm":
        load_audio_fn = LoadWebm()
    else:
        load_audio_fn = LoadWithTorchaudioDebug()

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        _ = (
            p
            | "Match Audio Files" >> beam_io.MatchFiles(known_args.match_pattern)
            | "Reshuffle Audio" >> beam.Reshuffle()
            | "SkipCompleted Audio" >> beam.ParDo(skip_completed)
            | "Filter Vocal Audio"
            >> beam.ParDo(
                FilterVocalAudio(known_args.audio_suffix, known_args.only_if_vocals)
            )
            | "Read Audio Matches" >> beam_io.ReadMatches()
            | "Load Audio" >> beam.ParDo(load_audio_fn)
            | "Crop Audio" >> beam.ParDo(CropAudioGTDuration(known_args.max_duration))
            | "Resample: 44.1k"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    target_sr=44_100,
                    source_sr_hint=48_000,
                )
            )
            | "SourceSeparate"
            >> beam.ParDo(
                SeparateSources(
                    source_dir=input_dir,
                    target_dir=input_dir,
                    audio_suffix=known_args.audio_suffix,
                    target_audio_suffix=TARGET_AUDIO_SUFFIX,
                    model_name="hdemucs_mmi",
                )
            )
            | "Resample: 48K"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    source_sr_hint=44_100,
                    target_sr=48_000,
                    output_numpy=True,
                )
            )
            | "CreateOGGFile" >> beam.Map(to_ogg)
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
