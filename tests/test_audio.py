import numpy as np


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
