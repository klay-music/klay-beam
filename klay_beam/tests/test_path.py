import pytest
from klay_beam.path import move, remove_suffix_pattern


def test_move_preserves_directory_structure():
    assert (
        move(
            "gs://audio/00/1009600.mp3",
            "gs://audio/",
            "/somewhere/else/",
        )
        == "/somewhere/else/00/1009600.mp3"
    ), "gs:// to local"

    assert (
        move(
            "/audio/00/1009600.mp3",
            "/audio/",
            "/somewhere/else/",
        )
        == "/somewhere/else/00/1009600.mp3"
    ), "local to local"

    assert (
        move(
            "/a//audio/00/1009600.mp3",
            "/a/audio//",
            "/somewhere/else/",
        )
        == "/somewhere/else/00/1009600.mp3"
    ), "local to local with extra slashes"

    assert (
        move(
            "/audio/00/1009600.mp3",
            "/audio/",
            "gs://somewhere/else/",
        )
        == "gs://somewhere/else/00/1009600.mp3"
    ), "local to gs://"


def test_move_fails():
    with pytest.raises(ValueError):
        move(
            "gs://audio/00/1009600.mp3",
            "gs://audio/01",
            "/somewhere/else/",
        )

    with pytest.raises(ValueError):
        move(
            "gs://audio/00/1009600.mp3",
            "//audio/00",
            "/somewhere/else/",
        )


@pytest.mark.parametrize(
    "path, pattern, exp",
    [
        ("path/to/audio.drums.wav", r".drums(-\d)?\.wav", "path/to/audio"),
        ("path/to/audio.drums.wav", ".drums.wav", "path/to/audio"),
        ("/path/to/audio.drums.wav", r".drums(-\d)?\.wav", "/path/to/audio"),
        ("gs://path/to/audio.drums.wav", r".drums(-\d)?\.wav", "gs://path/to/audio"),
        ("path/to/audio.drums-1.wav", r".drums(-\d)?\.wav", "path/to/audio"),
    ],
)
def test_remove_suffix_pattern(path, pattern, exp):
    got = remove_suffix_pattern(path, pattern)
    assert got == exp
