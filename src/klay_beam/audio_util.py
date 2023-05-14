import io
import pydub
import numpy as np

def numpy_to_mp3(audio_data: np.ndarray, sr: int, noisy=False):
    """Convert the audio tensor to an in-memory mp3 encoded file-like object.
    
    This needs to write a file-like object in memory. Loading file-like objects
    is supported by torchaudio (both sox_io and soundfile backends), but
    torchaudio.save can only write to disk. As a result, we use pydub, which
    invokes ffmpeg.

    For pytorch support see: https://pytorch.org/audio/stable/backend.html
    For pydub docs see: https://github.com/jiaaro/pydub
    """

    assert audio_data.ndim == 2, "audio_data must be 2 dimensional"
    samples, channels = audio_data.shape

    if noisy:
        print("Creating mp3: length: {} seconds. Channels: {}".format(samples / sr, channels))

    audio_data *= 2 ** 15 - 1
    audio_data = audio_data.mean(axis=1) # convert to mono
    audio_data = audio_data.astype(np.int16)
    
    raw_audio_buffer = io.BytesIO(audio_data.tobytes())

    audio_segment = pydub.AudioSegment.from_raw(raw_audio_buffer, frame_rate=sr, channels=1, sample_width=2)

    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    return mp3_buffer
