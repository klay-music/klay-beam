import pytest
import torch
import numpy as np

import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline as BeamTestPipeline
from apache_beam.testing.util import assert_that, equal_to

from job_klaynac.transforms import ExtractKlayNAC, CropAudioGTDuration


def test_ExtractKlayNAC():
    # Create an pseudo audio element with one second of 48kHz audio.
    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand((2, 48_000), dtype=torch.float32),
        48_000,
    )
    elements = [audio_element]

    # Run this on every element output by ExtractHFEncodec
    def output_validator(element):
        output_filename, tokens = element
        # assert len(tokens.shape) == 2

        # Is this the desired shape with time_steps last? That's
        # counterintuitive, but it does conform to our convention.
        tokens_per_step, time_steps = tokens.shape

        # Expect 16-bit signed integers
        assert tokens.dtype == np.int64

        # The discrete model should output 50 tokens for every second of audio, and
        # our input audio is one second long.
        assert time_steps == 50

        # The discrete model should output 24 codebook levels for every frame.
        assert tokens_per_step == 24

    with BeamTestPipeline() as p:
        extract_fn = ExtractKlayNAC()
        input_audio = p | beam.Create(elements)
        output_tokens = input_audio | beam.ParDo(extract_fn)

        # Verify that the length of the output collection is the same as the
        # length of the input collection
        assert_that(
            output_tokens | beam.combiners.Count.Globally(),
            equal_to([len(elements)]),
        )

        output_tokens | beam.Map(output_validator)


@pytest.mark.parametrize("audio_duration", [1, 10])
def test_CropAudioGTDuration_no_crop(audio_duration):
    max_duration = 10.0
    sr = 48_000
    transform = CropAudioGTDuration(max_duration)

    # Create an pseudo audio element with one second of 48kHz audio.
    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand((2, sr * audio_duration), dtype=torch.float32),
        sr,
    )

    _, x, _ = transform.process(audio_element)[0]
    assert x.shape[-1] == audio_element[1].shape[-1]


@pytest.mark.parametrize("audio_duration", [10.1, 20.0])
def test_CropAudioGTDuration_crop(audio_duration):
    max_duration = 10.0
    sr = 48_000
    transform = CropAudioGTDuration(max_duration)

    # Create an pseudo audio element with one second of 48kHz audio.
    audio_element = (
        "/absolute/path/to/source.wav",
        torch.rand((2, int(sr * audio_duration)), dtype=torch.float32),
        sr,
    )

    _, x, _ = transform.process(audio_element)[0]
    assert x.shape[-1] == max_duration * sr
