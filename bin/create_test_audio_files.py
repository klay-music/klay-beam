import numpy as np
from klay_beam.transforms import numpy_to_mp3, numpy_to_ogg, numpy_to_wav
import logging


def create_test_mono_audio_channel(fade_in_seconds: float = 1.0, sine=False):
    """Create a 2 second long mono audio data with white noise.

    Note the dimensionality. This is just a single channel in a single dimension
    (time) i.e. `[0, 0, 0, ...]`. Some functions in `klay_beam` may expect audio
    to be in a multi-channel format, (i.e. mono like this: `[[0, 0, 0, ...]]`).
    """
    sr = 44100
    seconds = 2
    total_samples = int(sr * seconds)
    audio_channel = np.random.rand(total_samples) * 2 - 1

    if sine:
        sine_frequency = 1000
        t = np.linspace(0, seconds, total_samples, endpoint=False)
        audio_channel = np.sin(2 * np.pi * sine_frequency * t)

    fade_in_samples = min(total_samples, int(sr * fade_in_seconds))
    fade_in = np.linspace(0.0, 1.0, fade_in_samples) ** 2.0

    audio_channel[:fade_in_samples] *= fade_in

    return audio_channel, sr


def create_test_audio_mono():
    mono_channel, sr = create_test_mono_audio_channel()
    return np.array([mono_channel * 0.25]), sr


def create_test_audio_stereo():
    left_channel, sr = create_test_mono_audio_channel(1.0)
    right_channel, _ = create_test_mono_audio_channel(2.0, sine=True)

    return np.array([left_channel * 0.25, right_channel * 0.25]), sr


def create_test_audio_4_channel():
    ch1, sr = create_test_mono_audio_channel(0.5)
    ch2, _ = create_test_mono_audio_channel(1.0, sine=True)
    ch3, _ = create_test_mono_audio_channel(1.5)
    ch4, _ = create_test_mono_audio_channel(2.0, sine=True)

    return np.array([ch1 * 0.25, ch2 * 0.25, ch3 * 0.25, ch4 * 0.25]), sr


def run():
    mono_audio, sr = create_test_audio_mono()
    mp3_buffer = numpy_to_mp3(mono_audio, sr)

    with open("test_audio/test_mono.mp3", "wb") as out_file:
        out_file.write(mp3_buffer.getvalue())
        mp3_buffer.seek(0)

    stereo_audio, sr = create_test_audio_stereo()
    mp3_buffer = numpy_to_mp3(stereo_audio, sr)

    with open("test_audio/test_stereo.mp3", "wb") as out_file:
        out_file.write(mp3_buffer.getvalue())
        mp3_buffer.seek(0)

    four_channel_audio, sr = create_test_audio_4_channel()
    found_error = False
    try:
        mp3_buffer = numpy_to_mp3(four_channel_audio, sr)
    except AssertionError:
        found_error = True
    assert (
        found_error
    ), "Expected an assertion error when passing 4 channel audio to numpy_to_mp3"

    with open("test_audio/test_stereo_16bit.wav", "wb") as out_file:
        out_file.write(numpy_to_wav(stereo_audio, sr, bit_depth=16).getvalue())

    with open("test_audio/test_stereo_24bit.wav", "wb") as out_file:
        out_file.write(numpy_to_wav(stereo_audio, sr, bit_depth=24).getvalue())

    with open("test_audio/test_stereo_32bit.wav", "wb") as out_file:
        out_file.write(numpy_to_wav(stereo_audio, sr, bit_depth=32).getvalue())

    with open("test_audio/test_stereo_64bit.wav", "wb") as out_file:
        out_file.write(numpy_to_wav(stereo_audio, sr, bit_depth=64).getvalue())

    with open("test_audio/test_quad.wav", "wb") as out_file:
        out_file.write(numpy_to_wav(four_channel_audio, sr).getvalue())

    with open("test_audio/test_stereo.ogg", "wb") as out_file:
        out_file.write(numpy_to_ogg(stereo_audio, sr, safe=False).getvalue())

    with open("test_audio/test_quad.ogg", "wb") as out_file:
        out_file.write(numpy_to_ogg(four_channel_audio, sr, safe=False).getvalue())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
