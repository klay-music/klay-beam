import logging
import apache_beam as beam
from klay_beam.transforms import numpy_to_wav

class SeparateSources(beam.DoFn):
    def __init__(self):
        pass

    def setup(self):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        pass

    def process(self, loaded_audio_tuple):
        key, audio_tensor, sr = loaded_audio_tuple

        in_memory_audio_file = numpy_to_wav(audio_tensor.numpy(), sr)
        return [(key+".44k1.wav", in_memory_audio_file)]
