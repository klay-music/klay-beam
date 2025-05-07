import apache_beam as beam
import apache_beam.io.fileio as beam_io
import av
import json
import logging
import numpy as np
import os.path
import io
from io import BytesIO
import torch
from typing import Tuple

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


class LoadWebm(beam.DoFn):
    """DoFn that turns a .webm audio file into (path, np.ndarray, sample_rate)."""

    @staticmethod
    def _load_webm(buf: bytes) -> tuple[np.ndarray, int]:
        """
        Decode a WebM/Opus byte blob → float32 numpy array (samples, channels).

        args:
            buf : bytes  WebM/Opus byte blob

        returns:
            audio : np.ndarray  (samples, channels)
            sr    : int         sample-rate reported by the stream
        """
        container = av.open(io.BytesIO(buf))
        stream = next(s for s in container.streams if s.type == "audio")

        # Fallback if metadata is missing
        sr = None
        if hasattr(stream, "rate") and stream.rate is not None:
            sr = stream.rate

        frames = (f.to_ndarray() for f in container.decode(stream))
        audio = np.concatenate(list(frames), axis=1).T.astype(np.float32)
        return audio, sr

    def process(self, readable_file: beam_io.ReadableFile):  # type: ignore
        path = Path(readable_file.metadata.path)
        logging.info(f"Loading {path}")

        try:
            with readable_file.open(mime_type="application/octet-stream") as f:
                data = f.read()

            audio, sr = self._load_webm(data)

            if sr is None:
                logging.warning("Missing sample rate for %s", path)
                return
        except Exception as exc:
            logging.error(f"Error decoding {path} : {exc}")
            return

        audio = np.transpose(audio)
        duration = audio.shape[1] / sr
        logging.info(
            f"Loaded {duration:.4f}s, {audio.shape[0]}-channel audio  ↪  {path}"
        )
        yield readable_file.metadata.path, audio, sr
