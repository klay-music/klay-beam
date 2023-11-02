import apache_beam as beam
from essentia.standard import TensorflowPredictEffnetDiscogs
import logging
import numpy as np
from typing import Tuple

from klay_beam.path import remove_suffix


class ExtractDiscogsEffnet(beam.DoFn):
    def setup(self, model_path: str):
        self.model = TensorflowPredictEffnetDiscogs(
            graphFilename=model_path, output="PartitionedCall:1"
        )

    @property
    def suffix(self):
        return ".discogs_effnet.npy"

    def process(
        self, audio_tuple: Tuple[str, np.ndarray, int]
    ) -> Tuple[str, np.ndarray, int]:
        fname, audio, sr = audio_tuple

        # validation
        assert sr == 16_000, f"DiscogsEffnet expects 16k audio. Found {sr}. ({fname})"
        if audio.ndim == 2 and audio.shape[0] == 1:
            audio = audio[0]
        elif audio.ndim == 1:
            pass
        else:
            raise ValueError(
                f"DiscogsEffnet expect 1D mono audio, got shape: {audio.shape}"
            )

        logging.info(f"Found audio file: {fname} with length: {len(audio)} samples.")

        # prepare file
        out_filename = remove_suffix(fname, ".wav")
        out_filename = remove_suffix(out_filename, ".mp3")
        out_filename += self.suffix

        # extract
        embeds = self.model(audio)
        embeds = np.transpose(embeds)
        return [(out_filename, embeds)]
