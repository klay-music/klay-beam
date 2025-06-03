import argparse
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)
import apache_beam as beam
import apache_beam.io.fileio as beam_io
import logging
import os
import os.path

from klay_beam.torch_transforms import (
    LoadWithTorchaudio,
)
from job_stats.transforms import GetStats, IsMusic, LoadNpy, GetGenre


DOCKER_IMAGE_NAME = os.environ.get("DOCKER_IMAGE_NAME", None)
if DOCKER_IMAGE_NAME is None:
    raise ValueError("Please set the DOCKER_IMAGE_NAME environment variable.")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_audio_path",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--match_suffix",
        default=".mp3",
        help="""
        Suffix to match audio files. Default is '.instrumental.stem.mp3'
        """,
    )

    return parser.parse_known_args(None)


def run():
    # Parse command line arguments
    known_args, pipeline_args = parse_args()
    logging.info("Got known arguments: {}".format(known_args))
    logging.info("Got pipeline arguments: {}".format(pipeline_args))

    # Pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Set the default docker image if we"re running on Dataflow
    if (
        pipeline_options.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_options.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_options.view_as(WorkerOptions).sdk_container_image = DOCKER_IMAGE_NAME

    # Pattern to recursively find audio files inside source_audio_path
    extract_fn = GetStats()

    # Patterns to recursively find files inside source_audio_path
    src_dir = known_args.source_audio_path.rstrip("/") + "/"
    match_pattern = src_dir + "**known_args.match_suffix"
    logging.info(f"match_pattern: {match_pattern}")

    yamnet_match_pattern = src_dir + "**.source.audioset_yamnet.npy"
    logging.info(f"yamnet match_pattern: {yamnet_match_pattern}")
    genre_discogs_match_pattern = src_dir + "**.source.genre_discogs400.npy"
    logging.info(f"genre_discogs match_pattern: {genre_discogs_match_pattern}")

    # Run the pipeline
    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        audio_stats_pcoll = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            | beam_io.ReadMatches()
            | "Load Audio Files" >> beam.ParDo(LoadWithTorchaudio())
            | "Run GetStats" >> beam.ParDo(extract_fn)
        )

        # fetch vocal classifier feature files and aggregate using IsVocalAudio
        is_music_pcoll = (
            p
            | "Match Yamnet Files" >> beam_io.MatchFiles(yamnet_match_pattern)
            | "Reshuffle Yamnet" >> beam.Reshuffle()
            | "Read Yamnet Matches" >> beam_io.ReadMatches()
            | "LoadNpy" >> beam.ParDo(LoadNpy())
            | "IsMusic" >> beam.ParDo(IsMusic())
        )

        genre_pcoll = (
            p
            | "Match Genre Files" >> beam_io.MatchFiles(genre_discogs_match_pattern)
            | "Reshuffle Genre" >> beam.Reshuffle()
            | "Read Genre Matches" >> beam_io.ReadMatches()
            | "LoadNpy Genre" >> beam.ParDo(LoadNpy())
            | "Get Genre" >> beam.ParDo(GetGenre())
        )

        # Count the total number of AudioStats objects
        total_count = (
            audio_stats_pcoll | "Count AudioStats" >> beam.combiners.Count.Globally()
        )

        # Sum the total duration of all AudioStats objects
        total_duration = (
            audio_stats_pcoll
            | "Extract Durations" >> beam.Map(lambda x: x.duration)
            | "Sum Durations" >> beam.CombineGlobally(sum)
        )

        # Count the number of stereo and mono audio
        stereo_count = (
            audio_stats_pcoll
            | "Extract Stereo Count"
            >> beam.Map(lambda x: ("stereo" if x.is_stereo else "mono", 1))
            | "Count Stereo and Mono" >> beam.combiners.Count.PerKey()
        )

        # Get sampling rate of audios in the collection
        distinct_sample_rates = (
            audio_stats_pcoll
            | "Extract Sampling Rates" >> beam.Map(lambda x: x.sr)
            | "Get Unique Sampling Rates" >> beam.Distinct()
        )

        # Get the total count of audios classified as music (or not)
        total_count_is_music = (
            is_music_pcoll
            | "Extract Is Music Count"
            >> beam.Map(lambda x: ("music" if x[1] else "not_music", 1))
            | "Count Is Music" >> beam.combiners.Count.PerKey()
        )

        total_count_is_genre = (
            genre_pcoll
            | "Get Genre Count"
            >> beam.FlatMap(
                lambda genre_tuple: ((genre, 1) for genre in genre_tuple[1])
            )
            | "Count Genres" >> beam.combiners.Count.PerKey()
        )

        # Output the results
        total_count | "Print Total Count" >> beam.Map(
            lambda x: logging.info(f"Total Count: {x}")
        )
        total_duration | "Print Total Duration" >> beam.Map(
            lambda x: logging.info(f"Total Duration: {x / 60 / 60} hours")
        )
        stereo_count | "Print Stereo Count" >> beam.Map(
            lambda x: logging.info(f"Stereo and Mono: {x}")
        )
        distinct_sample_rates | "Print Distinct Sample Rates" >> beam.Map(
            lambda x: logging.info(f"Distinct Sample Rates: {x}")
        )
        total_count_is_music | "Print Total Count (is Music)" >> beam.Map(
            lambda x: logging.info(f"Total is Music: {x}")
        )
        total_count_is_genre | "Print Total Count (is Genre)" >> beam.Map(
            lambda x: logging.info(f"Total Genre Count: {x}")
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
