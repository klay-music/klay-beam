import apache_beam as beam
import apache_beam.io.fileio as beam_io
import av
import io
import logging
import numpy as np
from pathlib import Path
import torch
from typing import Tuple, Optional

from klay_beam.path import remove_suffix
from klay_beam.utils import get_device
import klay_codecs
from klay_codecs.nac import KlayNAC, KlayNACVAE


class ExtractKlayNAC(beam.DoFn):
    """Beam DoFn for extracting KlayNAC tokens or KlayNACVAE embeds from audio."""

    def __init__(
        self,
        audio_suffix: str = ".mp3",
        device: Optional[torch.device] = None,
        extract_tokens: bool = True,
    ):
        self.audio_suffix = audio_suffix
        self._device = device
        self.extract_tokens = extract_tokens
        self.model_name = "klaynac" if extract_tokens else "klaynacvae"

    def setup(self):
        if self._device is None:
            self._device = get_device()

        if self.extract_tokens:
            self.nac = KlayNAC()
            self.nac.model.to(self._device)
            self.nac.model.eval()
            logging.info("Using KlayNAC")
        else:
            self.nac = KlayNACVAE()
            self.nac.model.to(self._device)
            self.nac.model.eval()
            logging.info(f"Using KlayNACVAE with config: {self.nac.config}")

    def __str__(self):
        return f"Extract-{self.model_name}"

    @property
    def output_file_format(self) -> str:
        return "npy"

    @property
    def suffix(self) -> str:
        return f".{self.model_name}-{klay_codecs.__version__}.npy"

    @property
    def num_channels(self) -> int:
        return self.nac.config.num_channels

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        assert source_sr == self.nac.config.sample_rate, (
            f"Source sample rate {source_sr} does not match the model's sample rate: {self.nac.config.sample_rate}."
        )

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, self.audio_suffix)
        output_filename += self.suffix
        logging.info(
            f"Processing audio with shape ({x.shape}): {output_filename} on {self._device}"
        )

        # normalize audio to [-1, 1]
        if x.abs().max() > 1:
            logging.info("Applying normalization")
            x = x / x.abs().max()

        audio_batch = x.unsqueeze(0).to(self._device)

        try:
            with torch.no_grad():
                if self.extract_tokens:
                    output_array = self.nac.audio_to_tokens(audio_batch)
                else:
                    output_array, _ = self.nac.audio_to_embeds(audio_batch)

                if self._device != torch.device("cpu"):
                    logging.info(
                        f"memory {torch.cuda.memory_allocated() / 1e9:.2f}GB / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                    )

                output_array = output_array.detach().cpu().numpy()
                if np.isnan(output_array).any():
                    logging.error(f"NaN values detected in {output_filename}")
                    return []
        except Exception as e:
            logging.error(
                f"{e} Failed to extract KlayNAC: {output_filename} of duration {x.shape[-1] / source_sr:.2f}s"
            )
            return []

        unbatched = output_array.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with KlayNAC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]


class CropAudioGTDuration(beam.DoFn):
    def __init__(self, max_duration: float):
        self.max_duration = max_duration

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        key, audio, sr = audio_tuple

        if audio.shape[-1] / sr > self.max_duration:
            logging.info(
                f"File is greater than {self.max_duration}s long. Cropping: {key}"
            )
            return [(key, audio[..., : int(self.max_duration * sr)], sr)]

        return [audio_tuple]


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
