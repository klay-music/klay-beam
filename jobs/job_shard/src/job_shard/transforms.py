import apache_beam as beam
from apache_beam.io.filesystems import FileSystems
from apache_beam.coders import VarIntCoder
from apache_beam.transforms import userstate
import logging
import os
from pathlib import Path

from klay_beam.path import remove_suffix


class FlattenTrack(beam.DoFn):
    """(track_name, Iterable[file_path]) → file_path"""

    def process(self, element, *args, **kwargs):
        _, files = element
        yield from files


class ShardCopy(beam.DoFn):
    """
    (shard_idx, List[audio_path]) → ( [src_dirs], [dst_paths] )

    Each *audio* path spawns one or more *feature* paths (one for every
    suffix).  The output tuples are consumed by CopyBinaryFileFn.
    """

    def __init__(
        self,
        *,
        src_dir: str,
        dest_dir: str,
        audio_suffix: str,
        suffixes: list[str],
        min_shard_idx: int = 1,
    ):
        if not suffixes:
            raise ValueError("suffixes must be non-empty")

        self.src_dir = src_dir if src_dir.endswith("/") else src_dir + "/"
        self.dest_dir = dest_dir if dest_dir.endswith("/") else dest_dir + "/"
        self.audio_suffix = audio_suffix
        self.suffixes = suffixes
        self.min_shard_idx = min_shard_idx

    def _rel_to_src(self, path: str) -> str:
        """Return path relative to the *source* directory."""
        return path[len(self.src_dir) :]

    def _dst_path(self, rel_path: Path, shard_idx: int) -> str:
        """Compose full destination URI for a given shard."""
        shard_idx += self.min_shard_idx
        shard_dir = f"shard-{shard_idx:05d}/"

        rel_source_path = rel_path.with_suffix(f".source{rel_path.suffix}")
        return os.path.join(self.dest_dir, shard_dir, rel_source_path)

    def process(self, element):
        shard_idx, batch = element

        for audio_path in batch:
            for suffix in self.suffixes:
                # Construct the feature path and check if it exists
                src_feat_path = (
                    remove_suffix(audio_path.path, self.audio_suffix) + suffix
                )
                if not FileSystems.match([src_feat_path]):
                    logging.warning("Feature file not found: %s", src_feat_path)
                    continue

                # Construct the destination path
                rel_path = self._rel_to_src(src_feat_path)
                dst_feat_path = self._dst_path(Path(rel_path), shard_idx)
                yield [src_feat_path], [dst_feat_path]


class _EnumerateDoFn(beam.DoFn):
    """
    Adds a monotonically-increasing integer to every element.
    Works in streaming or batch, on DirectRunner and Dataflow.
    """

    # One VALUE state cell per key, initialised to 0
    COUNTER = userstate.ReadModifyWriteStateSpec("c", VarIntCoder())

    def __init__(self, start: int = 0):
        self._start = start

    def process(
        self,
        element,
        counter_state=beam.DoFn.StateParam(COUNTER),
    ):
        # element comes from (None, element)
        _, elem = element

        current = counter_state.read() or self._start
        counter_state.write(current + 1)  # persist for the next element

        yield current, elem


class Enumerate(beam.PTransform):
    """(e1, e2, …) → (0, e1), (1, e2), …"""

    def __init__(self, start: int = 0):
        self._start = start

    def expand(self, pcoll):
        return (
            pcoll
            | "KeyByNone" >> beam.Map(lambda x: (None, x))
            | "AttachIndex" >> beam.ParDo(_EnumerateDoFn(self._start))
        )


class WriteManifest(beam.DoFn):
    """
    Writes MANIFEST.txt files for each shard containing video IDs.
    """

    def __init__(self, dest_dir: str, audio_suffix: str, min_shard_idx: int = 0):
        self.dest_dir = dest_dir if dest_dir.endswith("/") else dest_dir + "/"
        self.audio_suffix = audio_suffix
        self.min_shard_idx = min_shard_idx

    def _extract_video_id(self, file_path: str) -> str:
        """Extract video ID from file path.

        File structure: gs://<bucket_name>/shard-*/<video_id>/<video_id>.<suffix>
        """
        # Extract video ID from the directory structure
        path_parts = file_path.rstrip("/").split("/")
        if len(path_parts) >= 2:
            # The video ID is the second-to-last part of the path (parent directory)
            video_id = path_parts[-2]
            # Only use this if it's not empty and looks like a video ID
            if video_id and video_id != ".":
                return video_id

        # Fallback: extract from filename
        filename = os.path.basename(file_path)
        if filename.endswith(f".source{self.audio_suffix}"):
            video_id = filename[: -len(f".source{self.audio_suffix}")]
        else:
            video_id = (
                filename[: -len(self.audio_suffix)]
                if filename.endswith(self.audio_suffix)
                else filename
            )
        return video_id

    def process(self, element):
        shard_idx, files = element
        shard_idx += self.min_shard_idx
        shard_dir = f"shard-{shard_idx:05d}"

        # Extract video IDs from all files in this shard
        video_ids = set()
        for file_metadata in files:
            # file_metadata is a FileMetadata object with .path attribute
            file_path = file_metadata.path
            video_id = self._extract_video_id(file_path)
            video_ids.add(video_id)

        # Write MANIFEST.txt file
        manifest_path = f"{self.dest_dir}{shard_dir}/MANIFEST.txt"
        manifest_content = "\n".join(sorted(video_ids)) + "\n"

        try:
            with FileSystems.create(manifest_path) as f:
                f.write(manifest_content.encode("utf-8"))
            logging.info(
                f"Created MANIFEST.txt with {len(video_ids)} video IDs: {manifest_path}"
            )
        except Exception as e:
            logging.error(f"Failed to write MANIFEST.txt to {manifest_path}: {e}")

        yield element


class ReadManifestFiles(beam.DoFn):
    """Read MANIFEST.txt files and yield video IDs."""

    def process(self, element, *args, **kwargs):
        """Read a MANIFEST.txt file and yield video IDs."""
        manifest_path = element
        try:
            with FileSystems.open(manifest_path.path, "r") as f:
                content = f.read().decode("utf-8")
                video_ids = [
                    line.strip() for line in content.strip().split("\n") if line.strip()
                ]
                for video_id in video_ids:
                    yield video_id
                logging.info(
                    f"Read {len(video_ids)} video IDs from {manifest_path.path}"
                )
        except Exception as e:
            logging.warning(
                f"Failed to read MANIFEST.txt from {manifest_path.path}: {e}"
            )


class ExtractVideoIdFromFile(beam.DoFn):
    """Extract video ID from file path for keying."""

    def __init__(self, audio_suffix: str):
        self.audio_suffix = audio_suffix

    def _extract_video_id_from_path(self, file_path: str) -> str:
        """Extract video ID from file path.

        File structure: gs://<bucket_name>/path/<video_id>/<video_id>.<suffix>
        """
        # Extract video ID from the directory structure
        path_parts = file_path.rstrip("/").split("/")
        if len(path_parts) >= 2:
            # The video ID is the second-to-last part of the path (parent directory)
            video_id = path_parts[-2]
            # Only use this if it's not empty and looks like a video ID
            if video_id and video_id != ".":
                return video_id

        # Fallback: extract from filename
        filename = os.path.basename(file_path)
        video_id = (
            filename[: -len(self.audio_suffix)]
            if filename.endswith(self.audio_suffix)
            else filename
        )
        return video_id

    def process(self, element):
        """Extract video ID and key the file by it."""
        video_id = self._extract_video_id_from_path(element.path)
        yield (video_id, element)


class FilterOutExistingFiles(beam.DoFn):
    """Filter out files whose video IDs exist in the existing set."""

    def process(self, element):
        """Filter out files that have existing video IDs."""
        video_id, grouped_values = element
        files_list = list(grouped_values.get("files", []))
        existing_list = list(grouped_values.get("existing", []))

        # If no existing video ID found, yield all files
        if not existing_list:
            for file_metadata in files_list:
                yield file_metadata
        else:
            # Video ID exists, don't yield files
            logging.debug(f"Filtering out existing video ID: {video_id}")


class LoadExistingVideoIds(beam.PTransform):
    """PTransform to load all existing video IDs from MANIFEST.txt files."""

    def __init__(self, dest_dir: str):
        self.dest_dir = dest_dir.rstrip("/") + "/"

    def expand(self, pcoll):
        # Find all MANIFEST.txt files
        manifest_pattern = f"{self.dest_dir}shard-*/MANIFEST.txt"

        return (
            pcoll.pipeline
            | "MatchManifestFiles" >> beam.io.fileio.MatchFiles(manifest_pattern)
            | "ReadManifestFiles" >> beam.ParDo(ReadManifestFiles())
            | "KeyExistingVideoIds" >> beam.Map(lambda video_id: (video_id, True))
        )
