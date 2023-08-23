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
from klay_beam.transforms import remove_suffix


SAMPLE_RATE_MAP = {
    16000: "16khz",
    24000: "24khz",
    44100: "44khz",
    48000: "48khz",
}


class ExtractNAC(beam.DoFn):
    """Beam DoFn for extracting neural audio codec (NAC) tokens from audio."""

    def __init__(
        self,
        nac_name: str,
        sample_rate: int,
        device: torch.device = torch.device("cpu"),
    ):
        assert nac_name in [
            "encodec",
            "dac",
        ], f"Invalid name for neural audio codec: {nac_name}"
        self.nac_name = nac_name

        is_valid_sample_rate = True
        if nac_name == "encodec":
            is_valid_sample_rate = sample_rate in [24000, 48000]
        elif nac_name == "dac":
            is_valid_sample_rate = (sample_rate in [16000, 24000, 44100],)
        assert (
            is_valid_sample_rate
        ), f"Invalid sample_rate: {sample_rate} for neural audio codec: {nac_name}"

        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        if self.nac_name == "encodec":
            if self.sample_rate == 24000:
                self.codec = EncodecModel.encodec_model_24khz()
            elif self.sample_rate == 48000:
                self.codec = EncodecModel.encodec_model_48khz()
        elif self.nac_name == "dac":
            self.codec = DAC()
            self.codec = load_model(
                tag="latest", model_type=SAMPLE_RATE_MAP[self.sample_rate]
            )

        self.codec.eval()
        self.codec.to(self._device)

    @property
    def suffix(self) -> str:
        return f".{self.nac_name}_{SAMPLE_RATE_MAP[self.sample_rate]}.npy"

    @property
    def num_channels(self) -> int:
        return 2 if self.nac_name == "encodec" and self.sample_rate == 48000 else 1

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
            if self.nac_name == "encodec":
                # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
                # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
                codes = self.codec.encode(audio_batch)[0][0].numpy()
            elif self.nac_name == "dac":
                audio_signal = AudioSignal(
                    audio_batch, self.sample_rate, device=self._device
                )
                codes = encode(audio_signal, self._device, self.codec)["codes"]

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(
            f"Encoded with {self.nac_name.upper()} ({unbatched.shape}): {output_filename}"
        )
        return [(output_filename, unbatched)]
