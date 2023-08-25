from audiotools import AudioSignal
from dac.utils import load_model
from dac.model import DAC
from dac.utils.encode import process as encode
from encodec import EncodecModel
import logging
import torch
from typing import Optional, Tuple

import apache_beam as beam

from klay_data.transform import convert_audio
from klay_beam.path import remove_suffix


SAMPLE_RATE_MAP = {
    16000: "16khz",
    24000: "24khz",
    44100: "44khz",
    48000: "48khz",
}


class ExtractEncodec(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(self, sample_rate: int, device: torch.device = torch.device("cpu")):
        assert sample_rate in [
            24000,
            48000,
        ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        if self.sample_rate == 24000:
            self.codec = EncodecModel.encodec_model_24khz()
        elif self.sample_rate == 48000:
            self.codec = EncodecModel.encodec_model_48khz()

        self.codec.eval()
        self.codec.to(self._device)

    @property
    def suffix(self) -> str:
        return f".encodec_{SAMPLE_RATE_MAP[self.sample_rate]}.npy"

    @property
    def num_channels(self) -> int:
        return 2 if self.sample_rate == 48000 else 1

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav") + self.suffix

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.sample_rate,
            target_channels=self.num_channels,
        )
        audio_batch = audio.unsqueeze(0)

        with torch.no_grad():
            # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
            # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
            codes = self.codec.encode(audio_batch)[0][0].detach().cpu().numpy()

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with ENCODEC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]


class ExtractDAC(beam.DoFn):
    """Beam DoFn for extracting DAC tokens from audio."""

    def __init__(
        self,
        sample_rate: int,
        device: torch.device = torch.device("cpu"),
    ):
        assert sample_rate in [
            16000,
            24000,
            44100,
        ], f"Invalid sample_rate: {sample_rate} for ExtractDAC"

        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        self.codec = DAC()
        self.codec = load_model(
            tag="latest", model_type=SAMPLE_RATE_MAP[self.sample_rate]
        )
        self.codec.eval()
        self.codec.to(self._device)

    @property
    def suffix(self) -> str:
        return f".dac_{SAMPLE_RATE_MAP[self.sample_rate]}.npy"

    @property
    def num_channels(self) -> int:
        return 1

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav") + self.suffix

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.sample_rate,
            target_channels=self.num_channels,
        )
        audio_batch = audio.unsqueeze(0)

        with torch.no_grad():
            audio_signal = AudioSignal(
                audio_batch, self.sample_rate, device=self._device
            )
            codes = encode(audio_signal, self._device, self.codec)["codes"]

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with DAC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]
