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
    numpy_to_file,
)

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
    ResampleTorchaudioTensor,
)

from job_chroma.transforms import ExtractChromaFeatures


"""
Job for extracting Chroma features:
"""

# NOTE: the dependencies versions in Docker image must match the dependencies in
# the launch/dev environment. When updating dependencies, make sure that the
# docker image you specify for remote jobs also provides the correct
# dependencies. Here's where to look for each dependency.
#
# - pyproject.toml pins:
#   - apache_beam
#   - klay_beam
# - environment/dev.yml pins:
#   - pytorch
#   - python
#
# The default docker container specified in the bin/run_job_<name>.py script
# should provide identical dependencies.
DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.12.0-py3.10-beam2.51.0-torch2.0"


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
        "--stem",
        dest="stem",
        required=False,
        default=None,
        choices=["source", "bass", "drums", "other", "vocals"],
        help="The stem to extract",
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

    # Pattern to recursively find audio inside source_audio_path
    match_pattern = os.path.join(known_args.input, "**.wav")
    if known_args.stem is not None:
        match_pattern = os.path.join(known_args.input, f"**.{known_args.stem}.wav")

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
                    old_suffix=".wav",
                    # CAUTION! This if we change the chroma parameters, we need to change this too
                    new_suffix=".chroma_50hz.npy",
                )
            )
            # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
        )

        CHROMA_AUDIO_SR = 16_000

        (
            audio_files
            | f"Resample to F{CHROMA_AUDIO_SR}"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    source_sr_hint=48_000,
                    target_sr=CHROMA_AUDIO_SR,
                )
            )
            | "ExtractChroma"
            >> beam.ParDo(
                ExtractChromaFeatures(
                    # CAUTION! This if we change the chroma parameters, we need
                    # to also update the SkipCompleted transform.
                    audio_sr=CHROMA_AUDIO_SR,
                    n_chroma=12,
                    n_fft=2048,
                    win_length=1280,
                    hop_length=320,
                    norm=1,
                )
            )
            | "CreateNpyFile" >> beam.Map(lambda x: (x[0], numpy_to_file(x[1])))
            | "PersistFile" >> beam.Map(write_file)
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
