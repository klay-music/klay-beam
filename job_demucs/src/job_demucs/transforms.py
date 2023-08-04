import logging
import apache_beam as beam
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

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
        logging.info(f"Setting up DemucsSeparator: {self.model_name}")
        self.separator = DemucsSeparator(
            model_name=self.model_name,
            num_workers=1,
        )
        logging.info("DemucsSeparator setup")

    def process(self, loaded_audio_tuple):
        key, audio_tensor, sr = loaded_audio_tuple
        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({key})"

        out_filename = move(key, self.source_dir, self.target_dir)
        out_filename = out_filename.rstrip(".wav")
        out_filename = out_filename.rstrip(".source")

        logging.info(f"Separating: {key}")
        result_dict = self.separator(audio_tensor)
        pairs = [
            (f"{out_filename}.{k}.wav", numpy_to_wav(v, sr))
            for k, v in result_dict.items()
        ]

        for pair in pairs:
            logging.info(f"Separated: {pair[0]}")

        return pairs


class SkipCompleted(beam.DoFn):
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = source_dir
        self.target_dir = target_dir

    def process(self, file_metadata: FileMetadata):
        out_filename = move(file_metadata.path, self.source_dir, self.target_dir)

        if not out_filename.endswith(".source.wav"):
            logging.warn(f"Expected `.source.wav` extension: {file_metadata.path}")

        base_file_prefix = out_filename.rstrip(".source.wav")

        derived_files = [
            f"{base_file_prefix}.drums.wav",
            f"{base_file_prefix}.bass.wav",
            f"{base_file_prefix}.vocals.wav",
            f"{base_file_prefix}.other.wav",
        ]
        limits = [1, 1, 1, 1]

        logging.info(f"Checking if targets exist for: {file_metadata.path}")
        try:
            results = FileSystems.match(derived_files, limits=limits)
            for result in results:
                num_matches = len(result.metadata_list)
                # If any of the files do not exist, just forward the input
                logging.info(f"Found {num_matches} of: {result.pattern}")
                if num_matches != 1:
                    return [file_metadata]
            # At this point, we know a match was found for every file
            logging.info(f"Targets already exist. Skipping: {file_metadata.path}")
            return []
        except Exception as e:
            logging.warn(
                "Will not skip due to Exception while checking for target files. "
                + f"({file_metadata.path}). Exception: {e}"
            )
            return [file_metadata]
