import logging
import apache_beam as beam
from .paths import get_target_path
from .audio import random_crop
from klay_beam.transforms import numpy_to_wav


class Trim(beam.DoFn):
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = source_dir
        self.target_dir = target_dir

    def process(self, loaded_audio_tuple):
        """
        For a stereo audio file originally named '/path/to.some/file.key.mp3',
        and loaded with LoadWithTorchaudio expect:

        ```
        (
            '/path/to.some/file',
            'key.mp3',
            tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            44100
        )
        ```

        Transform that into a cropped audio file:
        ```
        (
            '/path/to/target_file.wav'
            in_memory_audio_file
        )
        ```
        """
        path, key, audio_tensor, sr = loaded_audio_tuple
        input_filename = f"{path}.{key}"
        target_filename = get_target_path(
            input_filename, self.source_dir, self.target_dir
        )
        cropped_audio_tensor = random_crop(audio_tensor.numpy(), sr)

        if cropped_audio_tensor is None:
            logging.warn(f"Audio file {input_filename} is too short. Skipping.")
            return []

        cropped_duration_seconds = cropped_audio_tensor.shape[1] / sr
        in_memory_wav = numpy_to_wav(cropped_audio_tensor, sr)
        logging.info(
            f"converted '{input_filename}' to a {cropped_duration_seconds:.3f} second .wav file"
        )

        return [(target_filename, in_memory_wav)]
