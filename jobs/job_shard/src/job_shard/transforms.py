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

    def process(self, kv):
        _, files = kv
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
        min_shard_idx: int = 0,
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
        shard_dir = f"shard-{shard_idx:04d}/"

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
        kv,
        counter_state=beam.DoFn.StateParam(COUNTER),
    ):
        # kv comes from (None, element)
        _, element = kv

        current = counter_state.read() or self._start
        counter_state.write(current + 1)  # persist for the next element

        yield current, element


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
