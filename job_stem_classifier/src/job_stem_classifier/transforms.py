import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.io.filesystems import FileSystems
from enum import StrEnum
import json
from pathlib import Path


class StemGroup(StrEnum):
    Bass = "bass"
    Drums = "drums"
    Other = "other"
    Vocals = "vocals"


class ClassifyAudioStem(beam.DoFn):
    def __init__(self, stem_map_path: Path):
        assert stem_map_path.is_file(), f"stem_map_path is not a file: {stem_map_path}"
        with open(stem_map_path, "r") as f:
            stem_map = json.load(f)

        # stem_map is a dict of the form: {"other": ["stem_name1", "stem_name2", ....]}
        # here we invert it to the form: {"stem_name1": "other", "stem_name2": "other", ...}
        self.stem_map = invert_stem_map(stem_map)

    def get_stem_group(self, stem_name: str) -> StemGroup:
        return self.stem_map[stem_name]

    def process(self, readable_file: beam_io.ReadableFile):
        orig_path = Path(readable_file.metadata.path)
        stem_name = parse_stem(orig_path.name)

        stem_group = self.stem_map.get(stem_name, None)
        if stem_group is None:
            new_path = orig_path
        else:
            suffix = f".{stem_group.value}{orig_path.suffix}"
            new_path = orig_path.parent / Path(orig_path.stem).with_suffix(suffix)

        return [(orig_path, new_path)]


def invert_stem_map(stem_map: dict[str, list[str]]) -> dict[str, StemGroup]:
    inverted_map = {}
    for stem_group, stem_list in stem_map.items():
        for stem in stem_list:
            inverted_map[stem] = StemGroup(stem_group)
    return inverted_map


def parse_stem(filename: str) -> str:
    return Path(filename).stem.split("_")[-1].lower()


def copy_file(source_dest_tup: tuple[Path, Path]):
    """Copy a file from source to destination. We use this to copy audio files
    based on their stem suffix, if a stem suffix already exists we will append
    a '-N' to the stem suffix where N is the next available natural number.
    """
    source, dest = source_dest_tup
    while FileSystems.exists(str(dest)):
        # parse the stem suffix from the dest path
        name_without_suffix = Path(dest).stem
        stem_suffix = Path(name_without_suffix).suffix
        parts = stem_suffix.split("-")

        # check whether the stem suffix has been incremented before
        # if not, increment it to 1 else increment the existing value by 1
        if len(parts) == 2:
            stem_name, count = parts
            new_suffix = f"{stem_name}-{int(count) + 1}{dest.suffix}"
        else:
            new_suffix = f"{stem_suffix}-1{dest.suffix}"

        # replace the stem suffix with the new one
        dest = dest.parent / Path(name_without_suffix).with_suffix(new_suffix)

    FileSystems.copy([str(source)], [str(dest)])
