import numpy as np
import torch
import torchaudio


# klay_data depends on Python > 3.9?
# copy pastaing this from klay-data/src/klay_data/utils.py
def klay_data_random_crop(x: np.ndarray, crop_length: int, dim: int = 0) -> np.ndarray:
    """Randomly crop a tensor along a given dimension.

    Parameters
    ----------
    x : np.ndarray
        Array to be cropped
    crop_length : int
        Length of the crop
    dim : int, optional
        Dimension along which to crop, by default 0

    Returns
    -------
    np.ndarray
        Cropped array
    """
    assert dim <= x.ndim, "dim must be less than x.ndim"
    if dim == 0:
        length = x.shape[0]
    elif dim == 1:
        length = x.shape[1]
    elif dim == 2:
        length = x.shape[2]
    if crop_length > length:
        # TODO: Should we pad instead?
        raise ValueError("crop length must be less than length of x")
    elif length == crop_length:
        return x

    start = np.random.randint(0, length - crop_length)
    end = start + crop_length

    if dim == 0:
        cropped = x[int(start) : int(end)]
    elif dim == 1:
        cropped = x[:, int(start) : int(end)]
    elif dim == 2:
        cropped = x[:, :, int(start) : int(end)]
    return cropped


# This is mostly a wrapper around klay_data's random_crop method.
def random_crop(
    audio_data: np.ndarray,
    sr: int,
    max_duration_seconds: float = 90,
    min_duration_seconds: float = 20,
):
    """
    Given a multi-channel audio ndarray, trim the audio to random 90 seconds.
    Return None if audio is shorter than 20 seconds
    """
    assert audio_data.ndim == 2, "audio_data shape must be (channels, samples))"

    channels, audio_duration_samples = audio_data.shape
    assert channels <= 128, "audio_data must have <= 128 channels"

    min_duration_samples = int(min_duration_seconds * sr)
    if audio_duration_samples < min_duration_samples:
        return None

    max_duration_samples = int(max_duration_seconds * sr)
    max_duration_samples = min(max_duration_samples, audio_duration_samples)

    return klay_data_random_crop(audio_data, max_duration_samples, dim=1)


# Copied from klay_data/src/klay_data/transform.py
def klay_data_convert_audio(
    wav: torch.Tensor, sr: int, target_sr: int, target_channels: int
):
    """Copied from encodec"""
    if wav.ndim == 1:
        wav.unsqueeze_(0)
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


# Wrapper around klay_data's convert_audio method, which
def ensure_audio_format(
    audio_data: np.ndarray, source_sr: int, target_sr: int, target_channels
):
    """
    Convert audio data to the specified sample rate and number of channels
    """
    resampled = klay_data_convert_audio(
        torch.from_numpy(audio_data), source_sr, target_sr, target_channels
    )
    return resampled.numpy()