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
    numpy_to_file,
    write_file,
)
from klay_beam.torch_transforms import LoadWithTorchaudio, ResampleTorchaudioTensor

from job_essentia_features.transforms import (
    ExtractEssentiaFeatures,
    ExtractEssentiaTempo,
    ALL_FEATURES,
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
        "--match_suffix",
        required=True,
        help="""
        The suffix of the files that we want to match.
        For example, .instrumental.stem.mp3
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        help="""
        The suffix of the audio files to process. For example, '.mp3'.

        Note that this suffix must be aligned with the audio suffix that's
        also provided by the user.
        """,
    )

    parser.add_argument(
        "--features",
        required=False,
        nargs="*",
        default=None,
        help=f"""
        The names of the essentia features that we want to extract and persist.
        Options: {ALL_FEATURES}
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

    # Pattern to recursively find audio files inside source_audio_path
    if known_args.audio_suffix not in known_args.match_suffix:
        raise ValueError(
            f"{known_args.match_suffix} does not contain {known_args.audio_suffix}"
        )

    src_dir = known_args.input.rstrip("/") + "/"
    match_pattern = src_dir + f"**{known_args.match_suffix}"
    extract_fn = ExtractEssentiaFeatures(known_args.audio_suffix, known_args.features)

    if known_args.features is None:
        return

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
                    new_suffix=list(extract_fn.suffixes.values()),
                    check_timestamp=True,
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
            | "Resample: 16k"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    target_sr=16_000,
                    source_sr_hint=48_000,
                )
            )
        )

        if known_args.features != ["tempo"]:
            # Extract Essentia features
            (
                audio_files
                | "ExtractEssentiaFeatures" >> beam.ParDo(extract_fn)
                # Write the classification result to .npy files in GCS
                | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
                | "PersistFile" >> beam.Map(write_file)
            )

        if "tempo" in known_args.features:
            # Extract tempo using TempoCNN
            (
                audio_files
                | "Resample to 11.25k for TempoCNN"
                >> beam.ParDo(
                    ResampleTorchaudioTensor(
                        target_sr=11_250,
                        source_sr_hint=16_000,
                    )
                )
                | "ExtractEssentiaTempo"
                >> beam.ParDo(ExtractEssentiaTempo(known_args.audio_suffix))
                | "CreateTempoNpyFile"
                >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
                | "PersistTempoFile" >> beam.Map(write_file)
            )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
