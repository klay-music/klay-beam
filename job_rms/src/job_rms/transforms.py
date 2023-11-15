import numpy as np
import librosa
import logging
from typing import Tuple
import apache_beam as beam

from klay_beam.path import remove_suffix


class ExtractRMSFeatures(beam.DoFn):
    """Extract RMS features from an audio tensor. We hard-code the extraction parameters
    because we don't really need to change them.
    """

    audio_sr = 16_000
    frame_rate = 50
    frame_length = 1280

    @property
    def feat_suffix(self):
        return f".rms_{self.frame_rate}hz.npy"

    @property
    def hop_length(self):
        if self.frame_rate % self.audio_sr == 0:
            raise ValueError(
                f"Audio sample rate({self.audio_sr}) must be a"
                f" multiple of frame rate ({self.frame_rate})"
            )
        return int(self.audio_sr / self.frame_rate)

    @staticmethod
    def extract_rms(
        audio: np.ndarray, hop_length: int, frame_length: int
    ) -> np.ndarray:
        """Extract RMS features from an audio tensor.

        Args:
            audio: 2D numpy.ndarray with audio in the last dimension
            hop_length: number of samples between successive frames
            frame_length: number of samples per frame
        Returns:
            2D numpy.ndarray with RMS features in the last dimension
        """
        return librosa.feature.rms(
            y=audio, hop_length=hop_length, frame_length=frame_length
        )

    def process(self, element: Tuple[str, np.ndarray, int]):
        """Extract RMS features from an audio tensor.

        An input `element` is deconstructed into 'filepath', 'audio', 'sr'.

        Args:
            key: The path of the audio file.
            audio: 2D numpy.ndarray with audio in the last dimension
            sr: sample rate of the audio
        Returns:
            A tuple of (key, features)
        """
        filepath, audio, sr = element

        try:
            features = self.extract_rms(
                audio=audio, hop_length=self.hop_length, frame_length=self.frame_length
            )
            output_path = remove_suffix(filepath, ".wav") + self.feat_suffix

            logging.info(
                f"Extracted RMS ({features.shape}) from audio ({audio.shape}): {output_path}"
            )

            return [(output_path, features)]

        except Exception as e:
            logging.error(f"Failed to extract chroma features for {filepath}: {e}")
            return []
