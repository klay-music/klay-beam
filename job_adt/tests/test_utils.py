import pytest


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
