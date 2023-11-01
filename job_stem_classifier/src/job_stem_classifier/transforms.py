import apache_beam as beam
import apache_beam.io.fileio as beam_io
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
        self.stem_map = self._invert_stem_map(stem_map)

    def get_stem_group(self, stem_name: str) -> StemGroup:
        return self.stem_map[stem_name]

    def process(self, readable_file: beam_io.ReadableFile):
        path = Path(readable_file.metadata.path)
        stem_name = self.parse_stem(path.name)

        stem_group = self.stem_map.get(stem_name, None)
        if stem_group is None:
            new_path = str(readable_file.metadata.path)
        else:
            suffix = f".{stem_group.value}{path.suffix}"
            new_path = str(path.parent / Path(path.stem).with_suffix(suffix))

        return [(readable_file.metadata.path, new_path)]


def _invert_stem_map(stem_map: dict[str, list[str]]) -> dict[str, StemGroup]:
    inverted_map = {}
    for stem_group, stem_list in stem_map.items():
        for stem in stem_list:
            inverted_map[stem] = StemGroup(stem_group)
    return inverted_map


def _parse_stem(filename: str) -> str:
    return Path(filename).stem.split("_")[-1]
