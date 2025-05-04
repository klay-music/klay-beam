import numpy as np
from pathlib import Path
import pytest

from job_demucs.transforms import (
    IsVocalAudio,
    create_new_filenames,
)


def test_create_new_filenames(tmp_path: Path):
    stem_groups = ["vocals", "drums", "other", "bass"]

    # case 1: simple filename input
    input_fp = str(tmp_path / "input.mp3")
    got = create_new_filenames(input_fp, stem_groups)
    expected = {
        "vocals": str(tmp_path / "input.vocals.wav"),
        "drums": str(tmp_path / "input.drums.wav"),
        "other": str(tmp_path / "input.other.wav"),
        "bass": str(tmp_path / "input.bass.wav"),
    }
    assert got == expected

    # case 2: processed filename input
    input_fp = str(tmp_path / "input.0.source.stem.mp3")
    got = create_new_filenames(input_fp, stem_groups)
    expected = {
        "bass": str(tmp_path / "input.1.bass.stem.wav"),
        "drums": str(tmp_path / "input.2.drums.stem.wav"),
        "other": str(tmp_path / "input.3.other.stem.wav"),
        "vocals": str(tmp_path / "input.4.vocals.stem.wav"),
    }
    assert got == expected

    # case 3: invalid filenames
    invalid_input_fps = [
        str(tmp_path / "input.stem.mp3"),
        str(tmp_path / "input.0.stem.mp3"),
        str(tmp_path / "input.0.source.mp3"),
        str(tmp_path / "input.source.mp3"),
    ]
    for input_fp in invalid_input_fps:
        pytest.raises(ValueError, create_new_filenames, input_fp, stem_groups)


def test_IsVocalAudio_majority_vote():
    probs = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.7]])
    p_threshold = 0.5
    n_threshold = 1.0

    got = IsVocalAudio.majority_vote(probs, p_threshold, n_threshold)

    # all items are above the threshold
    assert got

    probs = np.array([[0.3, 0.4, 0.7], [0.7, 0.6, 0.3]])
    got = IsVocalAudio.majority_vote(probs, p_threshold, n_threshold)

    # one item is equal to or below the threshold
    assert not got

    probs = np.array([[0.1, 0.2, 0.5], [0.9, 0.8, 0.5]])
    n_threshold = 0.6
    got = IsVocalAudio.majority_vote(probs, p_threshold, n_threshold)

    # 1/3 items is below the threshold ~= 0.66
    assert got

    n_threshold = 0.7
    got = IsVocalAudio.majority_vote(probs, p_threshold, n_threshold)
    # 1/3 items is below the threshold ~= 0.66
    assert not got


def test_IsVocalAudio(tmp_path: Path):
    # all frames are above the threshold -> is a vocal
    probs = np.array([[0.1, 0.2, 0.3, 0.4], [0.9, 0.8, 0.7, 0.6]])
    fp = str(tmp_path / "probs.npy")

    is_vocal_audio = IsVocalAudio()
    got_gen = is_vocal_audio.process((fp, probs))
    got_fp, got = next(got_gen)
    assert got_fp == fp
    assert isinstance(got_fp, str)
    assert got

    # 1 / 5 frames is below the threshold -> not a vocal
    probs = np.array([[0.1, 0.2, 0.3, 0.6], [0.9, 0.8, 0.7, 0.4]])
    got_gen = is_vocal_audio.process((fp, probs))
    got_fp, got = next(got_gen)
    assert got_fp == fp
    assert isinstance(got_fp, str)
    assert not got
