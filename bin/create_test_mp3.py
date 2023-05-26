import numpy as np
from klay_beam.transforms import numpy_to_mp3


def create_test_mono_audio_channel(fade_in_seconds: float = 1.0):
    """Create a 2 second long mono audio data with white noise.

    Note the dimensionality. This is just a single channel in a single dimension
    (time) i.e. `[0, 0, 0, ...]`. Some functions in `klay_beam` may expect audio
    to be in a multi-channel format, (i.e. mono like this: `[[0, 0, 0, ...]]`).
    """
    sr = 44100
    seconds = 2
    total_samples = int(sr * seconds)
    audio_channel = np.random.rand(total_samples) * 2 - 1

    fade_in_samples = min(total_samples, int(sr * fade_in_seconds))
    fade_in = np.linspace(0.0, 1.0, fade_in_samples) ** 2.0

    audio_channel[:fade_in_samples] *= fade_in

    return audio_channel, sr


def create_test_audio_mono():
    mono_channel, sr = create_test_mono_audio_channel()
    return np.array([mono_channel * 0.25]), sr


def create_test_audio_stereo():
    left_channel, sr = create_test_mono_audio_channel(1.0)
    right_channel, _ = create_test_mono_audio_channel(2.0)

    return np.array([left_channel * 0.25, right_channel * 0.25]), sr


def create_test_audio_4_channel():
    ch1, sr = create_test_mono_audio_channel(0.5)
    ch2, _ = create_test_mono_audio_channel(0.1)
    ch3, _ = create_test_mono_audio_channel(1.5)
    ch4, _ = create_test_mono_audio_channel(2.0)

    return np.array([ch1 * 0.25, ch2 * 0.25, ch3 * 0.25, ch4 * 0.25]), sr


def run():
    mono_audio, sr = create_test_audio_mono()
    mp3_buffer = numpy_to_mp3(mono_audio, sr, noisy=True)

    with open("test_mono.mp3", "wb") as out_file:
        out_file.write(mp3_buffer.getvalue())
        mp3_buffer.seek(0)

    stereo_audio, sr = create_test_audio_stereo()
    mp3_buffer = numpy_to_mp3(stereo_audio, sr, noisy=True)

    with open("test_stereo.mp3", "wb") as out_file:
        out_file.write(mp3_buffer.getvalue())
        mp3_buffer.seek(0)

    four_channel_audio, sr = create_test_audio_4_channel()
    found_error = False
    try:
        mp3_buffer = numpy_to_mp3(four_channel_audio, sr, noisy=True)
    except AssertionError:
        found_error = True
    assert (
        found_error
    ), "Expected an assertion error when passing 4 channel audio to numpy_to_mp3"


if __name__ == "__main__":
    run()
