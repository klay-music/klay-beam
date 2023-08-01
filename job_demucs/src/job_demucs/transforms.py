import logging
import apache_beam as beam


class SeparateSources(beam.DoFn):
    def __init__(self):
        pass

    def setup(self):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        pass

    def process(self, loaded_audio_tuple):
        return []
