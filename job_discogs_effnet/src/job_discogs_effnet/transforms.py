import apache_beam as beam
from essentia.standard import TensorflowPredictEffnetDiscogs
import librosa
import logging
import numpy as np
from pathlib import Path
from typing import Tuple

from .path import remove_suffix



class LoadWithLibrosa(beam.DoFn):
    """Use librosa to load audio files to numpy arrays.

    NOTES:

    Note that generally, custom functions have a few requirements that help them
    work well in on distributed runners. They are:
        - The function should be thread-compatible
        - The function should be serializable
        - Recommended: the function be idempotent

    For details about these requirements, see the Apache Beam documentation:
    https://beam.apache.org/documentation/programming-guide/#requirements-for-writing-user-code-for-beam-transforms
    """

    def __init__(self, target_sr: int):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        self.target_sr = target_sr

    def process(self, readable_file):
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, a, sr)` tuple where
            - `input_filename` is a string
            - `a` is a numpy array
            - `sr` is an int

        For a stereo audio file named '/path/to.some/file.key.mp3', return
        ```
        (
            '/path/to.some/file.key.mp3',
            np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            44100
        )
        ```
        """

        # I could not find good documentation for beam ReadableFile, so I'm
        # putting the key information below.
        #
        # ReadableFile Properties
        # - readable_file.metadata.path
        # - readable_file.metadata.size_in_bytes
        # - readable_file.metadata.last_updated_in_seconds
        #
        # ReadableFile Methods
        # - readable_file.open(mime_type='text/plain', compression_type=None)
        # - readable_file.read(mime_type='application/octet-stream')
        # - readable_file.read_utf8()

        path = Path(readable_file.metadata.path)

        # get the file extension without a period in a safe way
        ext_without_dot = path.suffix.lstrip(".")
        ext_without_dot = None if ext_without_dot == "" else ext_without_dot

        file_like = readable_file.open(mime_type="application/octet-stream")
        audio_array = None

        logging.info("Loading: {}".format(path))
        try:
            audio_array, sr = librosa.load(file_like, sr=self.target_sr, mono=True)
            assert sr == self.target_sr
        except RuntimeError:
            # We don't want to log the stacktrace, but for debugging, here's how
            # we could access it we can access it:
            #
            # import traceback
            # tb_str = traceback.format_exception(
            #     etype=type(e), value=e, tb=e.__traceback__
            # )
            logging.warning(f"Error loading audio: {path}")
            return (readable_file.metadata.path, np.ndarray([]), sr)

        logging.info(f"Loaded {len(audio_array) / sr:.3f} second mono audio: {path}")

        return [(readable_file.metadata.path, audio_array, sr)]


class ExtractDiscogsEffnet(beam.DoFn):
    
    model_path = str(Path(__file__).parents[2] / "models/discogs_multi_embeddings-effnet-bs64-1.pb")

    def setup(self):
        self.model = TensorflowPredictEffnetDiscogs(
            graphFilename=self.model_path, output="PartitionedCall:1"
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
