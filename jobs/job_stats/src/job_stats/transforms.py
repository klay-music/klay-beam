import apache_beam as beam
import apache_beam.io.fileio as beam_io
from dataclasses import dataclass
import logging
import numpy as np
import torch

from klay_beam.path import remove_suffix
from klay_data.feature_loader import AudiosetYamnetFeatureLoader
from job_stats.genre_classes import GENRE_CLASSES


@dataclass
class AudioStats:
    duration: float
    is_stereo: bool
    sr: int

    def __repr__(self):
        return f"AudioStats(duration={self.duration}, is_stereo={self.is_stereo}, sr={self.sr})"


class GetStats(beam.DoFn):
    """Beam DoFn for gathering dataset statistics."""

    def process(self, element: tuple[str, torch.Tensor, int]):
        filepath, audio, sr = element
        audio_stats = AudioStats(
            duration=audio.shape[1] / sr, is_stereo=audio.shape[0] == 2, sr=sr
        )
        logging.info(f"Got statistics: {audio_stats} for {filepath}")
        yield audio_stats


class IsMusic(beam.DoFn):
    """Compute whether the audio has speech or not."""

    def process(self, element: tuple[str, np.ndarray]):
        filepath, feats = element

        loader_cls = AudiosetYamnetFeatureLoader
        top_label, top_logit = loader_cls._get_top_label(feats)

        is_music = (
            top_label == loader_cls.music_class_label
            and top_logit >= loader_cls.music_prob_threshold
        )
        logging.info(f"{filepath} - is_music: {is_music}")
        yield filepath, is_music


class GetGenre(beam.DoFn):
    """Get the genre of the audio."""

    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold

    def process(self, element: tuple[str, np.ndarray]):
        filepath, feats = element

        genres = np.array(GENRE_CLASSES)[feats.max(axis=-1) > self.threshold].tolist()

        for genre in genres:
            logging.info(f"{filepath} - genre: {genre}")

            yield filepath, genre


class LoadNpy(beam.DoFn):
    """Load .npy files."""

    def process(self, readable_file: beam_io.ReadableFile):  # type: ignore
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, feats)` tuple where
            - `input_filename` is a string
            - `feats` is an np.ndarray
        """
        logging.info("Loading: {}".format(readable_file.metadata.path))
        with readable_file.open(mime_type="application/octet-stream") as file_like:
            feats = np.load(file_like)
        return [(readable_file.metadata.path, feats)]
