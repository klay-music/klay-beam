import logging

import torch
from typing import Tuple

import apache_beam as beam

from klay_beam.path import remove_suffix
from klay_beam.utils import get_device

from klay_data.extractors import MTRPPExtractor


class ExtractMTRPP(beam.DoFn):
    """Beam DoFn for extracting MTRPP embeddings from audio."""

    def __init__(
        self,
        audio_suffix: str,
        max_duration: float = 75.0,  # 60 seconds
    ):
        self.audio_suffix = audio_suffix
        self.max_duration = max_duration

    def setup(self):
        self._device = get_device()

        self.extractor = MTRPPExtractor(
            dummy_mode=False,
            device=self._device,
        )

    @property
    def suffix(self) -> str:
        return ".mtrpp.npy"

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = audio_tuple
        
        # handle the output filename
        output_filename = remove_suffix(key, self.audio_suffix) + self.suffix

        # check if the audio is longer than the max duration
        chunks = x.split(int(self.max_duration * source_sr), dim=-1)

        embeds = []
        for chunk in chunks:
            chunk = chunk.to(self._device)
            embeds.append(self.extractor(chunk, source_sr).squeeze(0))

        logging.info(f"Extracting MTRPP embeddings for {key} on {self._device}")
        embeds = torch.cat(embeds, dim=-1)  # [D, T]
        return [(output_filename, embeds.detach().cpu().numpy())]