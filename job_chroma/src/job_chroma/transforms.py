from typing import Tuple, Union
import logging
import torch
import apache_beam as beam

from klay_beam.torch_transforms import convert_audio
from klay_beam.path import remove_suffix

from .extractors import ChromaExtractor


class ExtractChromaFeatures(beam.DoFn):
    """Extract features from an audio tensor. Accepts a `(key, a, sr)` tuple
    were:

    - `key` is a string
    - `a` is a 2D torch.Tensor or numpy.ndarray with audio in the last dimension
    - `sr` is an int

    The return value will also be a `(key, features)` tuple
    """

    def __init__(
        self,
        audio_sr: int,
        # The default values below are just copied from the ChromaExtractor
        # on August 10, 2023. If the defaults change in the future, should
        # we change them in both places? It would be nice to find a way not
        # to maintain two copies of the same default values.
        n_chroma: int = 12,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: Union[int, None] = None,
        norm: float = torch.inf,
        device: Union["torch.device", str] = "cpu",
    ):
        self._audio_sr = audio_sr
        self._n_chroma = n_chroma
        self._n_fft = n_fft
        self._win_length = win_length
        self._hop_length = hop_length
        self._norm = norm
        self._device = device

    def setup(self):
        self._chroma_model = ChromaExtractor(
            sample_rate=self._audio_sr,
            n_chroma=self._n_chroma,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            norm=self._norm,
            device=self._device,
        )

    def process(self, element: Tuple[str, "torch.Tensor", int]):
        key, audio, sr = element

        try:
            # Ensure correct sample rate, and convert to mono
            audio = convert_audio(audio, sr, self._audio_sr, 1)

            features = self._chroma_model(audio)
            output_path = remove_suffix(key, ".wav") + self._chroma_model.feat_suffix

            logging.info(
                f"Extracted chroma ({features.shape}) from audio ({audio.shape}): {output_path}"
            )

            return [(output_path, features)]

        except Exception as e:
            logging.error(f"Failed to extract chroma features for {key}: {e}")
            return []
