from typing import Tuple
import logging
import time
import apache_beam as beam
import torchaudio
import torch

from klay_beam.transforms import convert_audio
from klay_beam.path import move, remove_suffix
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
        logging.info(f"Setting up DemucsSeparator: {self.model_name}")
        self.separator = DemucsSeparator(
            model_name=self.model_name,
            num_workers=1,
        )
        self.to_48k = torchaudio.transforms.Resample(44_100, 48_000)
        logging.info("DemucsSeparator setup")


    def process(self, loaded_audio_tuple: Tuple[str, torch.Tensor, int]):
        start_time = time.time()
        key, audio_tensor, sr = loaded_audio_tuple
        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({key})"
        channel_count, sample_count = audio_tensor.shape
        durationSeconds = sample_count / sr

        if channel_count != 2:
            logging.info(f"Converting from {channel_count} channel(s) to stereo: {key}")
            audio_tensor = convert_audio(audio_tensor, sr, 44_100, 2)

        out_filename = move(key, self.source_dir, self.target_dir)
        out_filename = remove_suffix(out_filename, ".wav")
        out_filename = remove_suffix(out_filename, ".source")

        logging.info(f"Separating: {key}")
        try:
            result_dict = self.separator(audio_tensor)

            triplets = [
                (f"{out_filename}.{k}.wav", v, sr)
                for k, v in result_dict.items()
            ]

            elapsed_time =  time.time() - start_time

            logging.info(
                "Separation complete! "
                f"Speed:{durationSeconds / elapsed_time:.3f}x realtime. "
                f"Audio Duration:{durationSeconds:.3f} seconds. "
                f"Processing time:{elapsed_time:.2f} seconds. "
                f"Key:{key}"
            )
            return triplets

        except Exception as e:
            logging.error(f"Exception while separating: {key}. Exception: {e}")
            return []


