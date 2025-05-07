import apache_beam as beam
import apache_beam.io.fileio as beam_io
import av
import io
import logging
import numpy as np
from pathlib import Path
import torch
import torchaudio
from typing import Tuple

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

    def _load_instrumental_audio(self, key: str) -> torch.Tensor:
        instrumental_audio_file = (
            remove_suffix(key, self.audio_suffix) + ".instrumental.mp3"
        )

        # load the instrumental audio file
        matches = beam.io.filesystems.FileSystems.match([instrumental_audio_file])[
            0
        ].metadata_list

        if not matches:
            logging.info(f"Skipping {instrumental_audio_file}, file not found.")
            return []

        readable_file = beam_io.ReadableFile(matches[0])

        with readable_file.open(mime_type="application/octet-stream") as file_like:
            audio, sr = torchaudio.load(file_like)

        return audio, sr

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = audio_tuple

        # TODO: Decide if we want to load instrumental audio or not
        # audio = self._load_instrumental_audio(key)

        # handle the output filename
        output_filename = remove_suffix(key, self.audio_suffix) + self.suffix

        # check if the audio is longer than the max duration
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).to(self._device)

        embeds = self.extractor(x, source_sr).squeeze(0)
        logging.info(f"Extracted MTRPP embeddings with shape {embeds.shape} for {key}")
        return [(output_filename, embeds.detach().cpu().numpy())]


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
