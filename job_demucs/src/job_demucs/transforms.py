import logging
import apache_beam as beam
from klay_beam.transforms import numpy_to_wav
from klay_beam.path import move


class SeparateSources(beam.DoFn):
    def __init__(self, source_dir: str, target_dir: str):
        self.source_dir = source_dir
        self.target_dir = target_dir
        pass

    def setup(self):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        pass

    def process(self, loaded_audio_tuple):
        key, audio_tensor, sr = loaded_audio_tuple
        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({key})"

        in_memory_audio_file = numpy_to_wav(audio_tensor.numpy(), sr)
        out_filename = move(key, self.source_dir, self.target_dir)

        logging.info(f"Completed: {out_filename}")
        return [(out_filename, in_memory_audio_file)]
