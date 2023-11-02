import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.io.filesystems import FileSystems
from enum import StrEnum
import json
import logging
import os
from pathlib import Path, PurePosixPath


assert os.path.sep == "/", "os.path.join (in get_target_path) breaks on Windows"


class StemGroup(StrEnum):
    Bass = "bass"
    Drums = "drums"
    Other = "other"
    Source = "source"
    Vocals = "vocals"


class ClassifyAudioStem(beam.DoFn):
    def __init__(self, stem_map_path: Path, source_dir: str, target_dir: str):
        self.stem_map_path = stem_map_path
        self.source_dir = source_dir
        self.target_dir = target_dir

    def setup(self):
        assert (
            self.stem_map_path.is_file()
        ), f"stem_map_path is not a file: {stem_map_path}"
        with open(self.stem_map_path, "r") as f:
            stem_map = json.load(f)

        # stem_map is a dict of the form: {"other": ["stem_name1", "stem_name2", ....]}
        # here we invert it to the form: {"stem_name1": "other", "stem_name2": "other", ...}
        self.stem_map = invert_stem_map(stem_map)

    def process(self, readable_file: beam_io.ReadableFile):
        orig_path = Path(readable_file.metadata.path)
        new_path, suffix = None, None

        if "_ST_" in orig_path.name:
            # Identify the stem group
            stem_name = parse_stem(orig_path.name)
            stem_group = self.stem_map.get(stem_name, None)

            if stem_group is not None:
                suffix = f".{stem_group.value}{orig_path.suffix}"
            else:
                logging.error(
                    f"stem_group is None for stem_name: {stem_name}. "
                    "This should not happen. Re-generate the stem_map file."
                )
        elif "_AM_" in orig_path.name:
            # master / instrumental, we classify these as "source"
            suffix = f".{StemGroup.Source.value}{orig_path.suffix}"
        elif "_BV_" in orig_path.name:
            # backing vocals
            suffix = f".{StemGroup.Vocals.value}{orig_path.suffix}"

        # NOTE: Here we are discarding the filename and using the directory / track
        # name as the filename. This is because we want all stems from the same track
        # to use the same name and be disambiguated by an enumerated stem group.
        if suffix is not None:
            parent = get_parent(readable_file.metadata.path)
            new_path = os.path.join(parent, Path(parent).stem + suffix)

        # finally, replace the source directory with the output directory
        if new_path is not None:
            new_path = replace_root_dir(new_path, self.source_dir, self.target_dir)

        return [(readable_file.metadata.path, new_path)]


def replace_root_dir(input_uri: str, source_dir: str, target_dir: str) -> str:
    """
    When processing datasets, we often have multi-layer directories, and we need to
    map input files to output files. Given an source directory and a target directory,
    map the input filenames to the output filenames, preserving the relative directory
    structure. This should work across local paths and GCS URIs.
    """
    input_path = PurePosixPath(input_uri)
    relative_filename = input_path.relative_to(source_dir)

    # pathlib does not safely handle `//` in URIs
    # `assert str(pathlib.Path("gs://data")) == "gs://data"` fails
    # As a result, we use os.path.join.
    return os.path.join(target_dir, relative_filename)


WILDCARD_CHARS = ["*", "?", "[", "]"]


def get_parent(path: str) -> str:
    """This is the equivalent of Path.parent but works on GCS URIs. It also
    strips out wildcard characters because gsutil doesn't like them.
    """
    parts = path.split("/")[:-1]
    parent = "/".join(parts)
    for char in WILDCARD_CHARS:
        parent = parent.replace(char, "")
    return parent


def invert_stem_map(stem_map: dict[str, list[str]]) -> dict[str, StemGroup]:
    inverted_map = {}
    for stem_group, stem_list in stem_map.items():
        for stem in stem_list:
            inverted_map[stem] = StemGroup(stem_group)
    return inverted_map


def parse_stem(filename: str) -> str:
    return Path(filename).stem.split("_")[-1].lower()


def copy_file(source_dest_tup: tuple[str, str | None]):
    """Copy a file from source to destination. We use this to copy audio files
    based on their stem suffix, if a stem suffix already exists we will append
    a '-N' to the stem suffix where N is the next available natural number.
    """
    source, dest = source_dest_tup
    if dest is None:
        return

    while FileSystems.exists(dest):
        # parse the stem suffix from the dest path
        name_without_suffix = Path(dest).stem
        stem_suffix = Path(name_without_suffix).suffix
        parts = stem_suffix.split("-")

        # check whether the stem suffix has been incremented before
        # if not, increment it to 1 else increment the existing value by 1
        if len(parts) == 2:
            stem_name, count = parts
            new_suffix = f"{stem_name}-{int(count) + 1}{Path(dest).suffix}"
        else:
            new_suffix = f"{stem_suffix}-1{Path(dest).suffix}"

        # replace the stem suffix with the new one
        dest = os.path.join(
            get_parent(dest), Path(name_without_suffix).with_suffix(new_suffix)
        )

    FileSystems.copy([str(source)], [str(dest)])
