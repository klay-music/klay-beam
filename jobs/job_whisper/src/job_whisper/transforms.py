import json
import logging
import os.path
import torch
from io import BytesIO
from typing import Tuple

import apache_beam as beam

from klay_data.extractors import WhisperExtractor
from klay_beam.utils import get_device


class ExtractWhisper(beam.DoFn):
    """Beam DoFn for extracting Whisper embeddings from audio.

    Args:
        device: torch.device, optional
            Device to use for extraction, by default None
    """

    def __init__(
        self,
        device: torch.device = None,
        vad_onset: float = 0.7,
        vad_offset: float = 0.4,
    ):
        self._device = device

        self.vad_onset = vad_onset
        self.vad_offset = vad_offset

    def setup(self):
        if self._device is None:
            self._device = get_device()
        logging.info(f"Using device: {self._device}")

        self.extractor = WhisperExtractor(
            device=self._device,
            vad_onset=self.vad_onset,
            vad_offset=self.vad_offset,
        )

    @property
    def suffix(self) -> str:
        return ".whisper.json"

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        try:
            key, x, source_sr = audio_tuple

            output_filename = os.path.splitext(key)[0]
            output_filename += self.suffix

            lyrics_dict: list[dict] = self.extractor(x, source_sr)

            if lyrics_dict:
                json_bytes = BytesIO(
                    json.dumps({"lyrics": lyrics_dict}).encode("utf-8")
                )
                logging.info(f"Extracted Whisper for {key}")
                return [(output_filename, json_bytes)]
            else:
                logging.info(f"No lyrics found for {key}")
                return []
        except Exception as e:
            logging.error(f"Exception while processing: {key}. Exception: {e}")
            return []
