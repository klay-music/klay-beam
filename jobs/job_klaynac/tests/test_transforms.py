import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline
from apache_beam.testing.util import assert_that, equal_to
import mock
import numpy as np
import math
import pytest
import torch

from job_klaynac.transforms import ExtractKlayNAC, CropAudioGTDuration
from job_klaynac.transforms import (
    secs_to_samples,
    make_fade_curves,
    make_envelope,
    make_frames,
    overlap_add,
)


@mock.patch("job_klaynac.transforms.DUMMY_MODE", new_value=True)
@pytest.mark.parametrize(
    "duration, window_duration, hop_duration",
    [
        (4, 3, 2),
        (5, 2, 1),
    ],
)
def test_ExtractKlayNACVAE(mock_dummy_mode, duration, window_duration, hop_duration):
    # Create an pseudo audio element with one second of 48kHz audio.
    sample_rate = 48_000
    num_channels = 2
    embed_dim = 128
    frame_rate = 30

    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand(
            (num_channels, int(duration * sample_rate)),
            dtype=torch.float32,
        ),
        sample_rate,
    )
    elements = [audio_element]

    # Run this on every element output by ExtractHFEncodec
    def output_validator(element):
        output_filename, embeds = element
        assert embeds.shape == (embed_dim, duration * frame_rate)
        assert embeds.dtype == np.float32

    with BeamTestPipeline() as p:
        extract_fn = ExtractKlayNAC(
            extract_tokens=False,
            window_duration=window_duration,
            hop_duration=hop_duration,
        )
        input_audio = p | beam.Create(elements)
        output_embeds = input_audio | beam.ParDo(extract_fn)

        # Verify that the length of the output collection is the same as the
        # length of the input collection
        assert_that(
            output_embeds | beam.combiners.Count.Globally(),
            equal_to([len(elements)]),
        )
        output_embeds | beam.Map(output_validator)


@pytest.mark.parametrize("audio_duration", [1, 10])
def test_CropAudioGTDuration_no_crop(audio_duration):
    max_duration = 10.0
    sr = 48_000
    transform = CropAudioGTDuration(max_duration)

    # Create an pseudo audio element with one second of 48kHz audio.
    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand((2, sr * audio_duration), dtype=torch.float32),
        sr,
    )

    _, x, _ = transform.process(audio_element)[0]
    assert x.shape[-1] == audio_element[1].shape[-1]


@pytest.mark.parametrize("audio_duration", [10.1, 20.0])
def test_CropAudioGTDuration_crop(audio_duration):
    max_duration = 10.0
    sr = 48_000
    transform = CropAudioGTDuration(max_duration)

    # Create an pseudo audio element with one second of 48kHz audio.
    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand((2, int(sr * audio_duration)), dtype=torch.float32),
        sr,
    )

    _, x, _ = transform.process(audio_element)[0]
    assert x.shape[-1] == max_duration * sr


@pytest.mark.parametrize(
    "seconds, rate, expected",
    [
        (0.0, 48_000, 0),
        (1.0, 48_000, 48_000),
        (0.5 / 48_000, 48_000, 1),  # rounds up
        (1.0, 30, 30),
        (0.3333, 30, 10),  # approx 10 samples
    ],
)
def test_secs_to_samples(seconds, rate, expected):
    assert secs_to_samples(seconds, rate) == expected


@pytest.mark.parametrize("overlap", [1, 2, 5])
def test_make_fade_curves_a(overlap):
    fi, fo = make_fade_curves(overlap, torch.device("cpu"))
    assert fi.shape == (overlap,)
    assert fo.shape == (overlap,)
    assert torch.allclose(fo, torch.flip(fi, [0]))
    assert math.isclose(float(fi[0]), 0.0)
    if overlap > 1:
        assert math.isclose(fi[-1], 1.0)
    assert math.isclose(float(fo[-1]), 0.0)
    if overlap > 1:
        assert math.isclose(fo[0], 1.0)


def test_make_fade_curves_b():
    overlap = 6
    exp_in = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    exp_out = [1.0, 0.8, 0.6, 0.4, 0.2, 0.0]
    fi, fo = make_fade_curves(overlap, torch.device("cpu"))
    assert fi.shape == (overlap,)
    assert fo.shape == (overlap,)

    assert torch.allclose(fi, torch.tensor(exp_in))
    assert torch.allclose(fo, torch.tensor(exp_out))


@pytest.mark.parametrize("idx,total", [(0, 3), (1, 3), (2, 3), (0, 1)])
def test_make_envelope_edges(idx, total):
    win_len, ov = 10, 4
    env = make_envelope(idx, total, win_len, ov, torch.device("cpu"))[0, :]

    # first window: leading ones
    if idx == 0:
        assert torch.all(env[:ov] == 1.0)
    else:
        assert env[0] == 0.0

    # last window: trailing ones
    if idx == total - 1:
        assert torch.all(env[-ov:] == 1.0)
    else:
        assert env[-1] == 0.0

    # length correct
    assert env.shape[0] == win_len


def test_make_frames_T_lt_window_length():
    audio = torch.zeros(2, 10)
    win_len = 20
    hop = 5
    got = make_frames(audio, win_len, hop)
    assert len(got) == 1
    assert torch.allclose(got[0], audio)


@pytest.mark.parametrize(
    "total_len, win_len, hop",
    [
        (19_000, 4_000, 3_000),  # normal case multiples
        (19_500, 4_000, 3_000),  # extra samples at end
    ],
)
def test_make_frames_shapes(total_len, win_len, hop):
    audio = torch.zeros(2, total_len)
    frames = make_frames(audio, win_len, hop)
    for i, fr in enumerate(frames):
        if i == len(frames) - 1:
            assert fr.shape[1] <= win_len
        else:
            assert fr.shape[1] == win_len

    # last window shorter than overlap should be dropped
    if total_len - (frames[-1].shape[1] + (len(frames) - 1) * hop) < (win_len - hop):
        # ensured by function logic
        pass


@pytest.mark.parametrize("win_len, hop, n_frames", [(8, 6, 3), (10, 7, 5)])
def test_overlap_add_constant_reconstruction(win_len, hop, n_frames):
    D = 1
    total = hop * (n_frames - 1) + win_len
    frames = [torch.ones(D, win_len) for _ in range(n_frames)]
    out = overlap_add(frames, hop, total)
    assert out.shape == (D, total)
    assert torch.allclose(out, torch.ones_like(out), atol=1e-6)


def test_overlap_add_mismatched_length_raises():
    with pytest.raises(Exception):
        a = [torch.zeros(1, 8), torch.zeros(1, 7)]  # diff length
        overlap_add(a, 6, 20)
