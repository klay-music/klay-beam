import apache_beam as beam
import apache_beam.io.fileio as beam_io
import json
import logging

from klay_beam.path import remove_suffix
from klay_beam.utils import get_device

from klay_data.extractors import ByT5TextExtractor
from klay_data.lyrics import WhisperLyrics


class LoadJson(beam.DoFn):
    def process(self, readable_file: beam_io.ReadableFile):
        with readable_file.open(mime_type="text/plain") as f:
            text = f.read().decode("utf-8")
        return [(readable_file.metadata.path, json.loads(text))]


class ExtractWhisperByT5(beam.DoFn):
    """Beam DoFn for extracting ByT5 tokens from text.

    More specifically, we use this DoFn to extract ByT5 tokens from
    JSON files extracted using whisperx. Besides extracting the embeddings,
    we also add the timing information to the output.
    """

    # we hardcode the whisper suffix because we only want to process whisper files
    whisper_suffix = ".whisper.json"

    def setup(self):
        self.device = get_device()
        logging.info(f"Using device: {self.device}")

        self.extractor = ByT5TextExtractor(self.device)

    @property
    def suffix(self) -> str:
        return ".whisper-byt5.npz"

    def process(self, element: tuple[str, dict]):
        filepath, lyrics_dict = element

        if not filepath.endswith(self.whisper_suffix):
            raise ValueError(
                f"Invalid file suffix: {filepath}, must end with {self.whisper_suffix}"
            )

        # remove and replace the suffix
        output_filepath = remove_suffix(filepath, self.whisper_suffix)
        output_filepath += self.suffix

        # extract tokens
        logging.info(f"Extracting ByT5 embeddings from {filepath}")
        lyrics = WhisperLyrics.from_dict(lyrics_dict)

        if len(lyrics) == 0:
            return

        embeds, tokens, start_array, end_array = lyrics.to_byt5(self.extractor)
        assert embeds.shape[-1] == len(tokens) == len(start_array) == len(end_array)

        yield output_filepath, {
            "byt5_embeds": embeds,
            "byt5_tokens": tokens,
            "starts": start_array,
            "ends": end_array,
        }
