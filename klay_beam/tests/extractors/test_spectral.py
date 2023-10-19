import librosa
import numpy as np

from klay_beam.torch_utils import TORCH_AVAILABLE, TORCH_IMPORT_ERROR
from ..utils import skip_if_no_torch

if TORCH_AVAILABLE:
    import torch
    from klay_beam.extractors.spectral import ChromaExtractor, CQTExtractor


@skip_if_no_torch
def get_sine(dur: int, sample_rate: int, freq: float) -> "torch.Tensor":
    return torch.sin(
        freq * 2 * torch.pi * torch.arange(sample_rate * dur) / sample_rate
    ).unsqueeze(0)


@skip_if_no_torch
def test_ChromaExtractor():
    sample_rate = 22050
    n_chroma = 12
    n_fft = 512
    win_length, hop_length = 512, 128

    extractor = ChromaExtractor(
        sample_rate,
        n_chroma=n_chroma,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
    )

    dur = 1
    freq = 440
    signal = get_sine(dur, sample_rate, freq)
    got = extractor(signal)

    exp_num_frames = int((sample_rate * dur) / hop_length) + 1
    assert got.shape == (n_chroma, exp_num_frames)

    # check that the "A" bin is maximal for each frame
    exp_bin = 9  # correspinging to A, bin 0 is C

    # ignore the first and last frames due to padding effects from the centered
    # spectrogram
    max_chroma_bins = got.argmax(dim=0)[1:-1].max()
    assert torch.allclose(max_chroma_bins, torch.tensor(exp_bin))

    extractor = ChromaExtractor(
        sample_rate,
        n_chroma=n_chroma,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        norm=1,
    )
    got = extractor(signal)

    exp_num_frames = int((sample_rate * dur) / hop_length) + 1
    assert got.shape == (n_chroma, exp_num_frames)
    assert torch.allclose(got.sum(dim=0).max(), torch.tensor(1.0))

    max_chroma_bins = got.argmax(dim=0)[1:-1].max()
    assert torch.allclose(max_chroma_bins, torch.tensor(exp_bin))


@skip_if_no_torch
def test_CQTExtractor():
    sample_rate = 22050
    n_bins = 72
    hop_length = 256

    extractor = CQTExtractor(
        sample_rate,
        n_bins=n_bins,
        hop_length=hop_length,
    )

    dur = 1
    freq = 440
    signal = get_sine(dur, sample_rate, freq)
    got = extractor(signal)

    exp_num_frames = int((sample_rate * dur) / hop_length) + 1
    assert got.shape == (n_bins, exp_num_frames)

    extractor = CQTExtractor(
        sample_rate,
        n_bins=n_bins,
        hop_length=hop_length,
        norm=1,
    )

    got = extractor(signal)

    exp_num_frames = int((sample_rate * dur) / hop_length) + 1
    assert got.shape == (n_bins, exp_num_frames)
    assert torch.allclose(got.sum(dim=-1).max(), torch.tensor(1.0))

    cqt_freqs = librosa.cqt_frequencies(n_bins, fmin=extractor.fmin)
    exp_max_bin = np.argmin(np.abs(cqt_freqs - freq))
    got_max_bins = got.argmax(dim=0)[1:-1].max()
    assert torch.allclose(got_max_bins, torch.tensor(exp_max_bin))
