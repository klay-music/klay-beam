import apache_beam as beam
import logging
import math
import numpy as np
import torch
from torch import Tensor
from typing import Tuple, Optional

from klay_beam.path import remove_suffix
from klay_beam.utils import get_device
import klay_codecs
from klay_codecs.nac import KlayNAC, KlayNACVAE


# Set to True to run in dummy mode (no GPU, no model loading)
DUMMY_MODE = False


class ExtractKlayNAC(beam.DoFn):
    """Beam DoFn for extracting KlayNAC tokens or KlayNACVAE embeds from audio."""

    def __init__(
        self,
        audio_suffix: str = ".mp3",
        device: Optional[torch.device] = None,
        extract_tokens: bool = True,
        window_duration: float = 205.0,
        hop_duration: float = 200.0,
    ):
        self.audio_suffix = audio_suffix
        self._device = device
        self.extract_tokens = extract_tokens
        self.window_duration = window_duration
        self.hop_duration = hop_duration
        self.model_name = "klaynac" if extract_tokens else "klaynacvae"

    def setup(self):
        if self._device is None:
            self._device = get_device()

        if self.extract_tokens:
            self.nac = KlayNAC()
            logging.info("Using KlayNAC")
        else:
            self.nac = KlayNACVAE(dummy_mode=DUMMY_MODE)
            logging.info(f"Using KlayNACVAE with config: {self.nac.config}")

        if hasattr(self.nac, "model"):
            self.nac.model.to(self._device)
            self.nac.model.eval()

    @property
    def audio_window_length(self) -> int:
        return secs_to_samples(self.window_duration, self.nac.config.sample_rate)

    @property
    def audio_hop_length(self) -> int:
        return secs_to_samples(self.hop_duration, self.nac.config.sample_rate)

    @property
    def embed_window_length(self) -> int:
        return secs_to_samples(self.window_duration, self.nac.config.frame_rate)

    @property
    def embed_hop_length(self) -> int:
        return secs_to_samples(self.hop_duration, self.nac.config.frame_rate)

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

    def process(self, element: Tuple[str, Tensor, int]):
        key, x, source_sr = element

        assert source_sr == self.nac.config.sample_rate, (
            f"Source sample rate {source_sr} does not match the model's"
            f" sample rate: {self.nac.config.sample_rate}."
        )

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, self.audio_suffix)
        output_filename += self.suffix
        if self._device != torch.device("cpu"):
            x = x.to(self._device)

        logging.info(
            f"Processing audio with shape ({x.shape}): {output_filename} on {self._device}"
        )

        audio_frames = make_frames(x, self.audio_window_length, self.audio_hop_length)
        embed_frames = []

        try:
            with torch.no_grad():
                for audio_frame in audio_frames:
                    if self.extract_tokens:
                        # NOTE: We've temporarily deactivated this because we are currently
                        # not supporting tokens also they don't work with overlap_add
                        # embeds = self.nac.audio_to_tokens(audio_frame.unsqueeze(0))
                        raise NotImplementedError(
                            "Token extraction is not supported. Use 'continuous' mode instead."
                        )
                    else:
                        embeds, _ = self.nac.audio_to_embeds(audio_frame.unsqueeze(0))

                    embed_frames.append(embeds[0].detach().cpu())

                if not embed_frames:
                    raise ValueError("No frames were extracted.")

                # Overlap-add the frames
                output_array = overlap_add(
                    embed_frames,
                    hop_length=self.embed_hop_length,
                    total_length=int(
                        ((x.shape[-1] / source_sr) * self.nac.config.frame_rate)
                    ),
                )

                if self._device != torch.device("cpu"):
                    logging.info(
                        f"memory {torch.cuda.memory_allocated() / 1e9:.2f}GB"
                        f" / {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
                    )

                output_array = output_array.detach().cpu().numpy()
                if np.isnan(output_array).any():
                    logging.error(f"NaN values detected in {output_filename}")
                    return []
        except Exception as e:
            logging.error(
                f"{e} Failed to extract KlayNAC: {output_filename} "
                f"of duration {x.shape[-1] / source_sr:.2f}s"
            )
            return []

        logging.info(f"Encoded with KlayNAC ({output_array.shape}): {output_filename}")
        return [(output_filename, output_array)]


class CropAudioGTDuration(beam.DoFn):
    def __init__(self, max_duration: float):
        self.max_duration = max_duration

    def process(self, audio_tuple: Tuple[str, Tensor, int]):
        key, audio, sr = audio_tuple

        if audio.shape[-1] / sr > self.max_duration:
            logging.info(
                f"File is greater than {self.max_duration}s long. Cropping: {key}"
            )
            return [(key, audio[..., : int(self.max_duration * sr)], sr)]

        return [audio_tuple]


def secs_to_samples(seconds: float, rate: int) -> int:
    return math.ceil(seconds * rate)


def make_frames(audio: Tensor, window_length: int, hop_length: int) -> list[Tensor]:
    """Slice audio into overlapping windows of size *hop*.

    The dimensions of each frame are (D, T), where T is the number of samples in the window.

    Args:
        audio: Tensor of shape (D, T) to be sliced into windows.
        hop: Size of each window in samples.
    """
    _, T = audio.shape

    if T < window_length:
        return [audio]

    starts = np.arange(0, T, hop_length).tolist()
    windows = list([audio[:, s : s + window_length] for s in starts])

    overlap = window_length - hop_length
    if windows[-1].shape[-1] < overlap:
        # If the last window is shorter than the overlap, we drop the last window
        windows = windows[:-1]
    return windows


def overlap_add(tensors: list[Tensor], hop_length: int, total_length: int) -> Tensor:
    """Linear overlap add along the *time* axis of tensors shaped (D, T) with asymmetric ramps.

    Args:
        tensors: List of tensors to be overlapped and added, each shaped (D, T).
        hop: Size of each window in samples.
    """
    if len(tensors) == 1:
        return tensors[0]

    D, window_length = tensors[0].shape
    overlap = window_length - hop_length
    out = torch.zeros(D, total_length, device=tensors[0].device)

    for idx, frame in enumerate(tensors):
        if frame.shape[-1] != window_length:
            envelope_length = frame.shape[-1]
        else:
            envelope_length = window_length

        envelope = make_envelope(
            idx, len(tensors), envelope_length, overlap, frame.device
        )
        start = idx * hop_length
        end = start + window_length
        out[:, start:end] += frame * envelope

    return out


def make_envelope(
    index: int, total: int, length: int, overlap: int, device: torch.device
) -> Tensor:
    """Asymmetric cross‑fade window according to position in sequence.

    Args:
        index: Index of the current frame in the sequence.
        total: Total number of frames in the sequence.
        length: Length of each frame.
        overlap: Length of the overlap region.
        device: Device on which to create the tensor.

    Returns:
        A tensor of shape (1, length) representing the envelope for the current frame.
    """
    fade_in, fade_out = make_fade_curves(overlap, device)
    w = torch.ones(length, device=device)

    if index > 0:
        # not first → ramp up
        w[:overlap] = fade_in
    if index < total - 1:
        # not last → ramp down
        w[-overlap:] = fade_out

    return w.unsqueeze(0)


def make_fade_curves(overlap: int, device: torch.device) -> tuple[Tensor, Tensor]:
    """Return linear fade‑in and fade‑out (1‑D) of length *overlap*."""
    fade_in = torch.linspace(0.0, 1.0, overlap, device=device)
    fade_out = torch.flip(fade_in, dims=[0])
    return fade_in, fade_out
