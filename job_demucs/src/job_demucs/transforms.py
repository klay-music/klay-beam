import logging
import apache_beam as beam
import numpy as np
from klay_beam.transforms import numpy_to_wav
from klay_beam.path import move

from job_demucs.demucs import DemucsSeparator


class SeparateSources(beam.DoFn):
    def __init__(
        self, source_dir: str, target_dir: str, model_name: str = "htdemucs_ft"
    ):
        """
        Arguments
        ---------
        source_dir
            The directory where the audio files are stored. Trim this from the
            source filename
        target_dir
            The directory where the output files should be stored.
        model_name
            Pretrained model name. The list of pre-trained models is:
                `mdx`: trained only on MusDB HQ, winning model on track A
                       at the MDX challenge.
                `mdx_extra`: trained with extra training data (including
                             MusDB test set). ranked 2nd on the track B of
                             the MDX challenge.
                `mdx_q`, `mdx_extra_q`: quantized version of the previous models.
                                        Smaller download and storage but quality
                                        can be slightly worse.
                `htdemucs_ft`: Fine tuned version of the default v4 model.
        """
        self.source_dir = source_dir
        self.target_dir = target_dir
        self.model_name = model_name

    def setup(self):
        # This will be executed only once when the pipeline starts.
        self.separator = DemucsSeparator(model_name=self.model_name)

    def process(self, loaded_audio_tuple):
        key, audio_tensor, sr = loaded_audio_tuple
        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({key})"

        out_filename = move(key, self.source_dir, self.target_dir)
        out_filename = out_filename.rstrip(".wav")
        out_filename = out_filename.rstrip(".source")

        result_dict = self.separator(audio_tensor)
        pairs = [
            (f"{out_filename}.{k}.wav", numpy_to_wav(v, sr))
            for k, v in result_dict.items()
        ]

        for pair in pairs:
            logging.info(f"Completed: {pair[0]}")

        return pairs
