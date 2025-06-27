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

from job_shard.transforms import (
    ShardCopy,
    Enumerate,
    WriteManifest,
    LoadExistingVideoIds,
    ExtractVideoIdFromFile,
    FilterOutExistingFiles,
)


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
            else:
                # Handle local file paths
                self._copy_local_file(source_path, dest_dir)

    def _copy_local_file(self, source_path, dest_path):
        """Copy a local file to another local path."""
        import shutil
        import os

        try:
            # Ensure destination directory exists
            dest_dir = os.path.dirname(dest_path)
            os.makedirs(dest_dir, exist_ok=True)

            # Copy the file
            shutil.copy2(source_path, dest_path)
            logging.debug(f"Copied {source_path} to {dest_path}")
        except Exception as e:
            logging.error(f"Error copying {source_path} to {dest_path}: {e}")


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

    if (
        known_args.max_dataset_size
        and known_args.max_dataset_size < known_args.num_files_per_shard
    ):
        known_args.num_files_per_shard = known_args.max_dataset_size

    # Note: We now allow existing shard files since we'll filter duplicates

    # Fully-qualified glob for MatchFiles
    src_root = known_args.src_dir.rstrip("/") + "/"
    match_pattern = src_root + f"**{known_args.audio_suffix}"

    with beam.Pipeline(argv=pipeline_args, options=pipeline_opts) as p:
        # -------------------------------------------------------------- #
        # 1. Find candidate audio files, filter out already processed ones
        # -------------------------------------------------------------- #
        audio_files = p | beam_io.MatchFiles(match_pattern) | beam.Reshuffle()

        # Filter out existing video IDs if destination directory is provided
        if known_args.dest_dir:
            # Load existing video IDs from MANIFEST.txt files
            existing_video_ids = audio_files | LoadExistingVideoIds(
                dest_dir=known_args.dest_dir
            )

            # Key audio files by video ID
            keyed_audio_files = audio_files | "KeyAudioFilesByVideoId" >> beam.ParDo(
                ExtractVideoIdFromFile(audio_suffix=known_args.audio_suffix)
            )

            # Join and filter out existing files
            audio_files = (
                {
                    "files": keyed_audio_files,
                    "existing": existing_video_ids,
                }
                | "CoGroupByVideoId" >> beam.CoGroupByKey()
                | "FilterOutExisting" >> beam.ParDo(FilterOutExistingFiles())
            )

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
        # 4. Write MANIFEST.txt files for each shard (if dest_dir provided)
        # -------------------------------------------------------------- #
        shard_with_manifest = batched
        if known_args.dest_dir:
            shard_with_manifest = batched | "WriteManifest" >> beam.ParDo(
                WriteManifest(
                    dest_dir=known_args.dest_dir,
                    audio_suffix=known_args.audio_suffix,
                    min_shard_idx=known_args.min_shard_idx,
                )
            )

        # -------------------------------------------------------------- #
        # 5. Build copy jobs and execute them
        # -------------------------------------------------------------- #
        _ = (
            shard_with_manifest
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
