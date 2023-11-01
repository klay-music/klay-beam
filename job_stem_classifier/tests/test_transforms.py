import pytest

from job_stem_classifier.transforms import _invert_stem_map, _parse_stem


def test__invert_stem_map():
    stem_map = {
        "other": ["guitar", "piano"],
        "drums": ["kick", "snare"],
    }
    got = _invert_stem_map(stem_map)
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
def test__parse_stem(filename, expected):
    got = _parse_stem(filename)
    assert got == expected
