import logging
import apache_beam as beam


class Dummy(beam.DoFn):
    def __init__(self):
        pass

    def process(self, loaded_audio_tuple):
        return []
