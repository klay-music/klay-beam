from pathlib import Path
import pytest
from tempfile import TemporaryDirectory

from job_stem_classifier.transforms import (
    invert_stem_map,
    parse_stem,
    copy_file,
    get_parent,
    replace_root_dir,
)


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
            copy_file((str(source_filepath), str(dest)))

            expected_dest = (
                Path(tmpdir) / f"some_guitar.{stem_name}-{max_count + i}.wav"
            )
            assert expected_dest.exists()


@pytest.mark.parametrize(
    "path, expected",
    [
        ("scratch/guitar.wav", "scratch"),
        ("guitar.wav", ""),
        ("/guitar.wav", ""),
        ("file://hello/guitar.wav", "file://hello"),
        ("gs://bucket/file/foo.bar.baz", "gs://bucket/file"),
    ],
)
def test_get_parent(path, expected):
    got = get_parent(path)
    assert got == expected


@pytest.mark.parametrize(
    "path, expected",
    [
        ("scratch?/guitar.wav", "scratch"),
        ("guitar.wav", ""),
        ("/guitar.wav", ""),
        ("file://hello[]/guitar.wav", "file://hello"),
        ("gs://bucket/file**/foo.bar.baz", "gs://bucket/file"),
    ],
)
def test_get_parent_with_wildcard_chars(path, expected):
    got = get_parent(path)
    assert got == expected


@pytest.mark.parametrize(
    "path, source_dir, root_dir, expected",
    [
        ("scratch/guitar.wav", "scratch", "data", "data/guitar.wav"),
        (
            "gs://scratch/guitar.wav",
            "gs://scratch",
            "gs://data",
            "gs://data/guitar.wav",
        ),
        (
            "/home/scratch/guitar.wav",
            "/home/scratch",
            "/home/data",
            "/home/data/guitar.wav",
        ),
    ],
)
def test_replace_root_dir(path, source_dir, root_dir, expected):
    got = replace_root_dir(path, source_dir, root_dir)
    assert got == expected


def test_replace_root_dir_fails():
    path = "gs://bucket/scratch/guitar.wav"
    source_dir = "scratch"
    root_dir = "data"
    with pytest.raises(ValueError):
        replace_root_dir(path, source_dir, root_dir)
