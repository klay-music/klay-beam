import pytest
import librosa
import numpy as np
import scipy

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


def spectral_divergence(a, b):
    """Compare the spectral magnitude of two signals. This effectively compares
    two signals in the frequency domain. The exact formula is from
    [Arik et al., 2018](https://arxiv.org/abs/1808.06719), but I've seen similar
    calculations used elsewhere.

    Why bother with a spectral comparison? Heuristic data compressors such as
    mp3 will make significant changes to the phase (and content) of a signal,
    but dominant frequency bands should be mostly preserved. We can use this to
    identify if (for example) our file writer replaced music with noise or
    silence. It is simple check that should also be robust against the small
    time shifts introduced by mp3 encoding.

    Empirically, two audio files with the same musical content, but
    different file encodings (e.g. wav vs mp3) will have a divergence of < 0.1.
    some examples:

    ```
    x_wav, _ = librosa.load("tests/test_data/music/01.wav", mono=False)
    x_mp3, _ = librosa.load("tests/test_data/music/01.mp3", mono=False)
    x_ogg, _ = librosa.load("tests/test_data/music/01.ogg", mono=False)
    x_opus, _ = librosa.load("tests/test_data/music/01.opus", mono=False)
    y_mp3, _ = librosa.load("tests/test_data/music/02.mp3", mono=False)

    # mp3 vs wav
    spectral_divergence(x_wav, x_mp3)
    [0.0395, 0.0385]

    # ogg vs wav
    spectral_divergence(x_wav, x_ogg)
    [0.0710, 0.0753]

    # different audio content
    spectral_divergence(x_wav, y_wav)
    [1.0358, 1.0477]
    ```

    However, it should be noted that these values will be affected by
    bitrate, the exact encoder used, and other parameters.

    Args:
        a, b: Two numpy arrays containing time-domain audio. Each array should
        have shape=(channels, samples). Note that this operation is not
        commutative (`spectral_divergence(a, b) != spectral_divergence(b, a)`).

    Returns:
        divergences: A numpy array of spectral divergence values, with one value
        for each channel. 0 means the signals are identical, 1 means they are
        quite different. The value represents an average across the entire
        signal (longer files will not have a higher value than shorter files).
    """
    assert a.shape == b.shape
    assert a.ndim == 2

    # apply an fft to each channel of each signal
    a_magnitudes = [np.abs(scipy.signal.stft(channel)[2]) for channel in a]
    b_magnitudes = [np.abs(scipy.signal.stft(channel)[2]) for channel in b]

    return np.array(
        [
            np.linalg.norm(a - b) / np.linalg.norm(a)
            for a, b in zip(a_magnitudes, b_magnitudes)
        ]
    )


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
    sine_b[: len(sine_b) // 2] += 0.3
    sine_b[len(sine_b) // 2 :] -= 0.3
    assert average_difference(sine_a, sine_b) == pytest.approx(0.3, 1e-14)

    # check it also works with stereo
    sine_a = np.array([sine_a, sine_a])
    sine_b = np.array([sine_b, sine_b])
    assert average_difference(sine_a, sine_a) == 0
    assert average_difference(sine_a, sine_b) == pytest.approx(0.3, 1e-14)


def test_spectral_divergence():
    sine_1k0, _ = create_test_sine(total_seconds=10, hz=1000)
    sine_1k1, _ = create_test_sine(total_seconds=10, hz=1100)
    sine_10k, _ = create_test_sine(total_seconds=10, hz=10000)

    sine_1k_stereo = np.stack([sine_1k0, sine_1k0])
    sine_1k1_stereo = np.stack([sine_1k1, sine_1k1])
    sine_10k_stereo = np.stack([sine_10k, sine_10k])

    noise_stereo = np.random.normal(size=sine_1k_stereo.shape)
    silence_stereo = np.zeros_like(sine_1k_stereo)

    # should be at least somewhat similar
    assert np.all(spectral_divergence(sine_1k_stereo, sine_1k_stereo) == 0)
    assert np.all(spectral_divergence(sine_1k_stereo, sine_1k1_stereo) < 0.5)

    # should be very different
    assert np.all(spectral_divergence(sine_1k_stereo, sine_10k_stereo) > 1.0)
    assert np.all(spectral_divergence(sine_1k_stereo, noise_stereo) > 1.0)
    assert np.all(spectral_divergence(sine_1k_stereo, silence_stereo) >= 1.0)


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


def test_mp3_file():
    sine, sr = create_test_sine()
    sine *= 0.5

    sine_mono = np.array([sine])
    sine_stereo = np.array([sine, sine])

    # make in-memory audio files
    mp3_mono = numpy_to_mp3(sine_mono, sr)
    mp3_stereo = numpy_to_mp3(sine_stereo, sr)

    # use Librosa to read the in-memory audio files. Annoyingly, librosa returns
    # mono audio as a 1-d array. As a result, we must compare to sine instead of
    # sine_mono
    librosa_mono, sr1 = librosa.load(mp3_mono, sr=None, mono=False)
    assert max_difference(sine, librosa_mono) < 1e-2
    assert average_difference(sine, librosa_mono) < 1e-4

    # Also verify the intermediary_bit_depth flag works as expected.
    mp3_mono = numpy_to_mp3(sine_mono, sr, intermediary_bit_depth=8)
    librosa_mono, sr1 = librosa.load(mp3_mono, sr=None, mono=False)
    assert max_difference(sine, librosa_mono) < 1e-1
    assert average_difference(sine, librosa_mono) < 1e-2

    # Try a stereo file
    librosa_stereo, sr2 = librosa.load(mp3_stereo, sr=None, mono=False)
    assert max_difference(sine_stereo, librosa_stereo) < 1e-2
    assert average_difference(sine_stereo, librosa_stereo) < 1e-4

    assert np.all(spectral_divergence(librosa_stereo, sine_stereo) < 0.1)

    # Double check that we are getting the same sample rate as we started with
    assert sr1 == sr
    assert sr2 == sr


def test_ogg_file():
    x, sr = librosa.load("tests/test_data/music/01.wav", sr=None, mono=False)
    in_memory_ogg = numpy_to_ogg(x, int(sr), safe=False)
    librosa_stereo, _ = librosa.load(in_memory_ogg, sr=None, mono=False)
    assert np.all(spectral_divergence(x, librosa_stereo) < 0.125)
