from pathlib import Path
import pytest
from tempfile import TemporaryDirectory

from job_stem_classifier.transforms import invert_stem_map, parse_stem, copy_file


def test_invert_stem_map():
    stem_map = {
        "other": ["guitar", "piano"],
        "drums": ["kick", "snare"],
    }
    got = invert_stem_map(stem_map)
    expected = {
        "guitar": "other",
        "piano": "other",
        "kick": "drums",
        "snare": "drums",
    }
    assert got == expected


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("guitar.wav", "guitar"),
        ("bla_piano.wav", "piano"),
        ("bla_.wav", ""),
        ("etc/bla_piano.wav", "piano"),
    ],
)
def test_parse_stem(filename, expected):
    got = parse_stem(filename)
    assert got == expected


def test_copy_file():
    stem_name = "other"
    with TemporaryDirectory() as tmpdir:
        # create source file
        source_filepath = Path(tmpdir) / "some_guitar.wav"
        source_filepath.touch()

        # create first dest file
        dest_files = []
        filepath = Path(tmpdir) / f"some_guitar.{stem_name}.wav"
        filepath.touch()
        dest_files.append(filepath)

        # create enumerated dest files
        max_count = 3
        for i in range(1, max_count):
            filepath = Path(tmpdir) / f"some_guitar.{stem_name}-{i}.wav"
            filepath.touch()
            dest_files.append(filepath)

        # now check that for each destination file we copy the source file
        # to a new file but with the enumeration after the stem suffix
        # incremented by 1 above the max
        for i, dest in enumerate(dest_files):
            # first make dest exists in the first place
            assert dest.exists()

            # copy file
            copy_file((source_filepath, dest))

            expected_dest = (
                Path(tmpdir) / f"some_guitar.{stem_name}-{max_count + i}.wav"
            )
            assert expected_dest.exists()
