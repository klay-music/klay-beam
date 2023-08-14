import logging
from typing import Tuple
import apache_beam as beam

from encodec import EncodecModel
from klay_data.transform import convert_audio
import torch


class ExtractEncodec(beam.DoFn):
    def __init__(self, device: torch.device = torch.device("cpu")):
        self._device = device

    def setup(self):
        self.encodec = EncodecModel.encodec_model_24khz().to(self._device)
        self.encodec.eval()

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        assert self.encodec.sample_rate == 24_000
        output_filename = key.rstrip(".wav") + ".encodec_24khz.npy"

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.encodec.sample_rate,
            target_channels=self.encodec.channels,
        )
        audio_batch = audio[None, :, :]

        with torch.no_grad():
            frames = self.encodec.encode(audio_batch)

        # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
        # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
        frame = frames[0]
        # If we initialize model with normalize=False (the default), I believe
        # we can discard the scale.
        codebook, _ = frame # `codebook` has shape `[B, K, T]`
        unbatched = codebook.squeeze(0) # `unbatched` has shape `[K, T]`

        logging.info(f"Encoded with Encodec ({unbatched.shape}): {output_filename}")

        return [(output_filename, unbatched.numpy())]
