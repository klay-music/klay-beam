import math
from librosa import cqt, filters
import numpy as np
from typing import Union

import torch
import torch.nn
import torch.nn.functional as F
from torchaudio.transforms import Spectrogram


class ChromaExtractor(torch.nn.Module):
    name = "chroma"

    def __init__(
        self,
        sample_rate: int,
        n_chroma: int = 12,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: Union[int, None] = None,
        norm: float = math.inf,
        device: Union["torch.device", str] = "cpu",
    ):
        super().__init__()

        self.device = device
        self.win_length = win_length
        self.n_fft = n_fft or self.win_length
        self.hop_length = hop_length or (self.win_length // 4)
        self.sr = sample_rate
        self.n_chroma = n_chroma
        self.norm = norm

        self.window = torch.hann_window(self.win_length).to(device)
        self.fbanks = torch.from_numpy(
            filters.chroma(
                sr=sample_rate, n_fft=self.n_fft, tuning=0, n_chroma=self.n_chroma
            )
        ).to(device)
        self.spec = Spectrogram(
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            power=2,
            center=True,
            pad=0,
            normalized=True,
        ).to(device)

    @property
    def audio_sr(self) -> int:
        return self.sr

    @property
    def feat_sr(self) -> int:
        return self.sr // self.hop_length

    @property
    def feat_suffix(self) -> str:
        return f".{self.name}_{self.feat_sr}hz.npy"

    @staticmethod
    def _pad(audio: "torch.Tensor", n_fft: int) -> "torch.Tensor":
        T = audio.shape[-1]

        # in case we are getting a audio that was dropped out (nullified)
        # make sure audio length is no less that n_fft
        if T < n_fft:
            pad = n_fft - T
            r = 0 if pad % 2 == 0 else 1
            audio = F.pad(audio, (pad // 2, pad // 2 + r), "constant", 0)
            assert (
                audio.shape[-1] == n_fft
            ), f"expected len {n_fft} but got {audio.shape[-1]}"
        return audio

    def forward(self, audio: "torch.Tensor"):
        audio = self._pad(audio, self.n_fft)
        spec = self.spec(audio).squeeze(1)
        raw_chroma = torch.einsum("cf, ...f t-> ...ct", self.fbanks, spec)
        norm_chroma = F.normalize(raw_chroma, p=self.norm, dim=-2, eps=1e-6)

        # returns features in shape: [d, t]
        return norm_chroma.squeeze(0)


class CQTExtractor(torch.nn.Module):
    name = "cqt"

    def __init__(
        self,
        sample_rate: int,
        hop_length: int,
        n_bins: int = 12,
        fmin: float = 32.7,
        norm: float = torch.inf,
    ):
        super().__init__()

        self.sr = sample_rate
        self.n_bins = n_bins
        self.hop_length = hop_length
        self.fmin = fmin
        self.norm = norm

    @property
    def audio_sr(self) -> int:
        return self.sr

    @property
    def feat_sr(self) -> int:
        return self.sr // self.hop_length

    def forward(self, audio: "torch.Tensor") -> "torch.Tensor":
        spec = np.abs(
            cqt(
                audio.detach().cpu().numpy(),
                sr=self.sr,
                hop_length=self.hop_length,
                fmin=self.fmin,
                n_bins=self.n_bins,
            )
        )
        spec = torch.from_numpy(spec).squeeze(0)
        norm_spec = F.normalize(spec, p=self.norm, dim=-1, eps=1e-6)

        # returns features in shape: [d, t]
        return norm_spec.squeeze(0)

    @property
    def feat_suffix(self) -> str:
        return f".{self.name}_{self.feat_sr}hz.npy"
