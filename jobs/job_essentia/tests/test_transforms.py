from pathlib import Path
import pytest
import torch

from job_essentia.transforms import (
    ExtractEssentiaFeatures,
    ExtractEssentiaTempo,
)


EXPECTED_FEATURE_SHAPES = {
    "genre_discogs400": (400, 9),
    "mtg_jamendo_genre": (87, 9),
    "approachability": (2, 9),
    "danceability": (2, 9),
    "engagement": (2, 9),
    "mood_aggressive": (2, 9),
    "mood_happy": (2, 9),
    "mood_party": (2, 9),
    "mood_relaxed": (2, 9),
    "mood_sad": (2, 9),
    "mtg_jamendo_moodtheme": (56, 9),
    "mtg_jamendo_instrument": (40, 9),
    "mood_acoustic": (2, 9),
    "mood_electronic": (2, 9),
    "voice_instrumental": (2, 9),
    "timbre": (2, 9),
    "nsynth_instrument": (11, 9),
    "nsynth_reverb": (2, 9),
    "tonal_atonal": (2, 9),
    "mtg_jamendo_top50tags": (50, 9),
    "mtt": (50, 9),
}


@pytest.mark.parametrize("audio_suffix", [".mp3", ".flac"])
def test_ExtractEssentiaFeatures(tmp_path: Path, audio_suffix: str):
    extract_fn = ExtractEssentiaFeatures(audio_suffix)
    extract_fn.setup()

    sr = 16_000
    duration = 10.0
    audio = torch.rand(1, int(duration * sr))
    audio_path = str(tmp_path / f"test{audio_suffix}")

    got = extract_fn.process((audio_path, audio, sr))
    for output in got:
        feat_name = output[0].split(".")[-2]
        assert (
            "." + ".".join(Path(output[0]).name.split(".")[1:])
            in extract_fn.suffixes.values()
        )
        assert output[1].shape == EXPECTED_FEATURE_SHAPES[feat_name]
        assert output[1].ndim == 2


@pytest.mark.parametrize("audio_suffix", [".mp3", ".flac"])
def test_ExtractEssentiaTempo(tmp_path: Path, audio_suffix: str):
    extract_fn = ExtractEssentiaTempo(audio_suffix)
    extract_fn.setup()

    sr = 11_025
    duration = 10.0
    audio = torch.rand(1, int(duration * sr))
    audio_path = str(tmp_path / f"test{audio_suffix}")

    got = extract_fn.process((audio_path, audio, sr))
    for output in got:
        assert output[0].endswith(".tempo.npy")
        assert output[1].shape == (1, 1)
        assert output[1].ndim == 2
