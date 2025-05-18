import argparse
import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.transforms.util import BatchElements
from apache_beam.io.fileio import FileMetadata
from apache_beam.io.filesystems import FileSystems
from apache_beam.io.filesystem import BeamIOError
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)
import logging
import os

from klay_beam.transforms import SkipCompleted
from job_shard.transforms import ShardCopy, Enumerate


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--src_dir",
        required=True,
        help="""
        Specify the parent directory. This can be a local path or a gs:// URI.
        """,
    )

    parser.add_argument(
        "--dest_dir",
        type=str,
        help="""
        Specify the parent directory where the output files will be written.
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        help="""
        Which audio file extension is being used? This is also the
        audio file extension that will be replaced by the new
        feature file extension.
        """,
    )

    parser.add_argument(
        "--suffixes",
        required=True,
        nargs="+",
        help="""
        List of suffixes of the files that should be copied over.
        """,
    )

    parser.add_argument(
        "--num_files_per_shard",
        type=int,
        default=100000,
        help="""
        Number of files to be copied per shard. Default is 1,000,000.
        """,
    )

    parser.add_argument(
        "--max_dataset_size",
        type=int,
        default=None,
        help="""
        Maximum number of files to process. Default is None (process all).
        """,
    )

    parser.add_argument(
        "--min_shard_idx",
        type=int,
        default=0,
        help="""
        Minimum shard index to start from. Default is 0.
        """,
    )

    return parser.parse_known_args(None)


def copy_files(element):
    source_paths, destination_paths = element
    try:
        FileSystems.copy(source_paths, destination_paths)
    except BeamIOError as e:
        logging.error(f"Error copying {source_paths} to {destination_paths}: {e}")


def read_write_file(source_path, dest_path):
    try:
        # Read entire file from GCS
        with FileSystems.open(source_path, "rb") as src_file:
            data = src_file.read()

        # Write entire file to S3
        with FileSystems.create(dest_path) as dst_file:
            dst_file.write(data)

        logging.debug(f"Copied {source_path} to {dest_path}.")
    except Exception as e:
        logging.error(f"Error copying {source_path} to {dest_path}: {e}")


class CopyBinaryFileFn(beam.DoFn):
    def process(self, element):
        source_paths, dest_dirs = element

        for source_path, dest_dir in zip(source_paths, dest_dirs):
            if source_path.startswith("gs://") and dest_dir.startswith("gs://"):
                copy_files(([source_path], [dest_dir]))
            elif source_path.startswith("s3://") and dest_dir.startswith("s3://"):
                copy_files(([source_path], [dest_dir]))
            elif source_path.startswith("gs://") and dest_dir.startswith("s3://"):
                read_write_file(source_path, dest_dir)
            elif source_path.startswith("s3://") and dest_dir.startswith("gs://"):
                read_write_file(source_path, dest_dir)


def get_dir_name(md: FileMetadata) -> tuple[str, str]:
    """(dir_name, full_path)"""
    track_name = md.path.split("/")[-2]
    return track_name, md.path


CHUNK = 1_000  # work-unit size; tweak for your cluster


def chunk(batch, chunk_size=CHUNK):
    """[f0, …, f99999] -> [[f0…f999], [f1000…], …]"""
    for i in range(0, len(batch), chunk_size):
        yield batch[i : i + chunk_size]


def run():
    known_args, pipeline_args = parse_args()

    pipeline_opts = PipelineOptions(pipeline_args)
    pipeline_opts.view_as(SetupOptions).save_main_session = True

    if (
        pipeline_opts.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_opts.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_opts.view_as(WorkerOptions).sdk_container_image = os.environ[
            "DOCKER_IMAGE_NAME"
        ]

    if known_args.max_dataset_size < known_args.num_files_per_shard:
        known_args.num_files_per_shard = known_args.max_dataset_size

    # Check if a shard already exists in the destination directory
    if known_args.dest_dir:
        shard_files = FileSystems.match([f"{known_args.dest_dir}/shard-*"])
        if shard_files:
            raise ValueError(
                "Destination directory already contains shard files. "
                "Please choose a different destination directory."
            )

    # Fully-qualified glob for MatchFiles
    src_root = known_args.src_dir.rstrip("/") + "/"
    match_pattern = src_root + f"**{known_args.audio_suffix}"

    with beam.Pipeline(argv=pipeline_args, options=pipeline_opts) as p:
        # -------------------------------------------------------------- #
        # 1. Find candidate audio files, skip the ones already copied
        # -------------------------------------------------------------- #
        audio_files = p | beam_io.MatchFiles(match_pattern) | beam.Reshuffle()

        # -------------------------------------------------------------- #
        # 2. Optionally cap the data set size (first N elements)
        # -------------------------------------------------------------- #
        if known_args.max_dataset_size:
            audio_files = (
                audio_files
                | "IndexAll" >> Enumerate()  # (idx, path)
                | "KeepFirstN"
                >> beam.Filter(
                    lambda kv, limit: kv[0] < limit,
                    limit=known_args.max_dataset_size,
                )
                | "DropIdx" >> beam.Map(lambda kv: kv[1])  # back to path
            )

        # -------------------------------------------------------------- #
        # 3. Batch into fixed-width shards
        # -------------------------------------------------------------- #
        batched = (
            audio_files
            | "BatchIntoShards"
            >> BatchElements(
                min_batch_size=known_args.num_files_per_shard,
                max_batch_size=known_args.num_files_per_shard,
            )
            | "AddShardIdx" >> Enumerate()  # (shard_idx, big_batch)
            | "ExplodeIntoChunks"  # (idx, small_batch)
            >> beam.FlatMap(lambda kv, sz=CHUNK: ((kv[0], c) for c in chunk(kv[1], sz)))
            | "ShardReshuffle"
            >> beam.Reshuffle()  # keep (idx, big_batch) but randomise keys
        )

        # -------------------------------------------------------------- #
        # 4. Build copy jobs and execute them
        # -------------------------------------------------------------- #
        _ = (
            batched
            | "BuildCopyPairs"
            >> beam.ParDo(
                ShardCopy(
                    src_dir=src_root,
                    dest_dir=known_args.dest_dir,
                    audio_suffix=known_args.audio_suffix,
                    suffixes=known_args.suffixes,
                    min_shard_idx=known_args.min_shard_idx,
                )
            )
            | "CopyFiles" >> beam.ParDo(CopyBinaryFileFn())
        )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
