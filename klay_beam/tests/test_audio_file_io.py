import pytest
import librosa
import numpy as np

from klay_beam.transforms import (
    numpy_to_mp3,
    numpy_to_ogg,
    numpy_to_wav,
)


def create_test_sine(
    hz: float = 1000,
    sr: int = 44100,
    total_seconds: float = 0.5,
    fade_in_seconds: float = 0,
):
    """Create a 0.5 second mono audio signal with a sine wave

    Note the dimensionality. This is just a single channel in a single dimension
    (time) i.e. `[0, 0, 0, ...]`. Some functions in `klay_beam` may expect audio
    to be in a multi-channel format, (i.e. mono like this: `[[0, 0, 0, ...]]`).
    """

    total_samples = int(sr * total_seconds)
    fade_in_samples = int(sr * fade_in_seconds)
    fade_in_samples = min(total_samples, fade_in_samples)

    t = np.linspace(0, total_seconds, total_samples, endpoint=False)
    audio_channel = np.sin(2 * np.pi * hz * t)

    fade_in = np.linspace(0.0, 1.0, fade_in_samples) ** 2.0
    audio_channel[:fade_in_samples] *= fade_in

    return audio_channel, sr


def total_difference(a, b):
    # calculate the difference between two numpy arrays, returning a scalar, no
    # matter what the input rank is
    assert a.shape == b.shape
    return np.sum(np.abs(a - b))


def max_difference(a, b):
    assert a.shape == b.shape
    return np.max(np.abs(a - b))


def average_difference(a, b):
    assert a.shape == b.shape
    return np.average(np.abs(a - b).reshape(-1))


def test_sine():
    sine, _ = create_test_sine(hz=1, sr=4, total_seconds=1)
    expected_sine = np.array([0, 1, 0, -1])

    assert sine.shape == (4,)
    assert total_difference(sine, expected_sine) < 1e-14


def test_total_difference():
    sine_a, _ = create_test_sine()
    sine_b, _ = create_test_sine()

    # Make sine_b different from sine_a by a total of 1.0
    sine_b[0] -= 0.5
    sine_b[1] += 0.5

    assert total_difference(sine_a, sine_a) == 0
    assert total_difference(sine_a, sine_b) == 1.0


def test_max_difference():
    sine_a, _ = create_test_sine()
    sine_b, _ = create_test_sine()

    # Make sine_b different from sine_a by a total of 1.0
    sine_b[0] += 0.3
    sine_b[1] -= 0.75

    assert max_difference(sine_a, sine_a) == 0
    assert max_difference(sine_a, sine_b) == pytest.approx(0.75, 1e-14)
    sine_b[2] += 0.8
    assert max_difference(sine_a, sine_b) == pytest.approx(0.8, 1e-14)


def test_average_difference():
    sine_a, _ = create_test_sine()
    sine_b, _ = create_test_sine()

    sine_b -= 0.3

    assert average_difference(sine_a, sine_a) == 0
    assert average_difference(sine_a, sine_b) == pytest.approx(0.3, 1e-14)

    # Check that we are getting the average of the absolute differences
    sine_b, _ = create_test_sine()
    sine_b[:len(sine_b) // 2] += 0.3
    sine_b[len(sine_b) // 2:] -= 0.3
    assert average_difference(sine_a, sine_b) == pytest.approx(0.3, 1e-14)

    # check it also works with stereo
    sine_a = np.array([sine_a, sine_a])
    sine_b = np.array([sine_b, sine_b])
    assert average_difference(sine_a, sine_a) == 0
    assert average_difference(sine_a, sine_b) == pytest.approx(0.3, 1e-14)


def test_wav_file():
    sine, sr = create_test_sine()
    sine *= 0.5

    sine_mono = np.array([sine])
    sine_stereo = np.array([sine, sine])

    # make in-memory audio files
    wav_mono = numpy_to_wav(sine_mono, sr)
    wav_stereo = numpy_to_wav(sine_stereo, sr)

    # use Librosa to read the in-memory audio files. Annoyingly, librosa returns
    # mono audio as a 1-d array. As a result, we must compare to sine instead of
    # sine_mono
    librosa_mono, sr1 = librosa.load(wav_mono, sr=None, mono=False)
    assert max_difference(sine, librosa_mono) < 1e-4
    assert average_difference(sine, librosa_mono) < 1e-4

    # Do the same check, but verify that it works with 24-bit audio. Note that
    # we expect a higher precision from 24-bit audio.
    wav_mono = numpy_to_wav(sine_mono, sr, bit_depth=24)
    librosa_mono, sr1 = librosa.load(wav_mono, sr=None, mono=False)
    assert max_difference(sine, librosa_mono) < 1e-6
    assert average_difference(sine, librosa_mono) < 1e-6

    # Try a stereo file
    librosa_stereo, sr2 = librosa.load(wav_stereo, sr=None, mono=False)
    assert max_difference(sine_stereo, librosa_stereo) < 1e-4
    assert average_difference(sine_stereo, librosa_stereo) < 1e-4

    # Double check that we are getting the same sample rate as we started with
    assert sr1 == sr
    assert sr2 == sr
