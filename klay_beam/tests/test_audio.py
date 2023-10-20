import numpy as np
from klay_beam.torch_transforms import ResampleTorchaudioTensor

from .utils import skip_if_no_torch
from klay_beam.torch_utils import TORCH_AVAILABLE


if TORCH_AVAILABLE:
    import torch


def test_interleave():
    """Verify that we can use numpy's .T property to interleave stereo audio"""

    # This is how stereo audio is normally stored with the rightmost dimension
    # containing audio arrays ready for DSP.
    deinterleaved_audio = np.array(
        [[10, 20, 30, 40, 50, 60, 70, 80, 90], [-1, -2, -3, -4, -5, -6, -7, -8, -9]]
    )

    # However, when passing audio to pydub, it needs to be interleaved, that is,
    # stored like this
    expected_result = np.array(
        [
            [10, -1],
            [20, -2],
            [30, -3],
            [40, -4],
            [50, -5],
            [60, -6],
            [70, -7],
            [80, -8],
            [90, -9],
        ]
    )

    actual_result = deinterleaved_audio.T

    assert np.array_equal(
        actual_result, expected_result
    ), f"expected {actual_result} to equal {expected_result}"


def test_multichannel_interleave():
    """Verify that we can use numpy's .T property to interleave multi-channel audio"""

    deinterleaved_audio = np.array(
        [
            [10, 20, 30, 40, 50, 60, 70, 80, 90],
            [-1, -2, -3, -4, -5, -6, -7, -8, -9],
            [50, 50, 50, 50, 50, 50, 50, 50, 50],
        ]
    )

    expected_result = np.array(
        [
            [10, -1, 50],
            [20, -2, 50],
            [30, -3, 50],
            [40, -4, 50],
            [50, -5, 50],
            [60, -6, 50],
            [70, -7, 50],
            [80, -8, 50],
            [90, -9, 50],
        ]
    )

    actual_result = deinterleaved_audio.T

    assert np.array_equal(
        actual_result, expected_result
    ), f"expected {actual_result} to equal {expected_result}"


@skip_if_no_torch
def test_resample():
    """Verify that we can resample both numpy and torch audio (to a torch.Tensor))"""
    stereo_audio_tensor = ("key1", torch.randn(2, 1000), 1000)
    stereo_audio_numpy = ("key2", stereo_audio_tensor[1].numpy(), 1000)

    resampleDoFn = ResampleTorchaudioTensor(target_sr=10, source_sr_hint=1000)

    _, resampled_stereo_audio_tensor, _ = resampleDoFn.process(stereo_audio_tensor)[0]
    assert resampled_stereo_audio_tensor.shape == (
        2,
        10,
    ), "Failed to resample torch.tensor to sr=10"
    assert isinstance(
        resampled_stereo_audio_tensor, torch.Tensor
    ), "Failed to return a torch.Tensor"

    _, resampled_stereo_audio_numpy, _ = resampleDoFn.process(stereo_audio_numpy)[0]
    assert resampled_stereo_audio_numpy.shape == (
        2,
        10,
    ), "Failed to resample ndarray to sr=10"
    assert isinstance(
        resampled_stereo_audio_numpy, torch.Tensor
    ), "Failed to return a torch.tensor"


@skip_if_no_torch
def test_resample_to_numpy():
    """Verify that we can resample both numpy and torch audio (to a numpy.ndarray)"""
    stereo_audio_tensor = ("key1", torch.randn(2, 1000), 1000)
    stereo_audio_numpy = ("key2", stereo_audio_tensor[1].numpy(), 1000)

    resampleDoFn = ResampleTorchaudioTensor(target_sr=10, output_numpy=True)

    _, resampled_stereo_audio_tensor, _ = resampleDoFn.process(stereo_audio_tensor)[0]
    assert resampled_stereo_audio_tensor.shape == (
        2,
        10,
    ), "Failed to resample torch.tensor to sr=10"
    assert isinstance(
        resampled_stereo_audio_tensor, np.ndarray
    ), "Failed to return a torch.ndarray"

    _, resampled_stereo_audio_numpy, _ = resampleDoFn.process(stereo_audio_numpy)[0]
    assert resampled_stereo_audio_numpy.shape == (
        2,
        10,
    ), "Failed to resample ndarray to sr=10"
    assert isinstance(
        resampled_stereo_audio_numpy, np.ndarray
    ), "Failed to return a np.ndarray"
