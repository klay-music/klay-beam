import logging
from huggingface_hub import hf_hub_download
import laion_clap
from pathlib import Path
import torch
from typing import Tuple

import apache_beam as beam

from klay_beam.path import remove_suffix
from klay_beam.utils import get_device


REPO_ID = "lukewys/laion_clap"
CACHE_DIR = Path.home() / ".cache/huggingface/hub/models--lukewys--laion_clap/snapshots"
FILENAME = "music_audioset_epoch_15_esc_90.14.pt"


def download_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
    return model_path


class ExtractCLAP(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    sample_rate = 48000
    frame_duration = 5.0

    def __init__(self):
        self.num_samples_in_frame = int(self.sample_rate * self.frame_duration)

    def setup(self):
        self._device = get_device()

        cached_models = list(CACHE_DIR.glob(f"**/{FILENAME}"))
        if not cached_models:
            logging.info("Downloading model from HuggingFace Hub")
            model_path = download_model()
        else:
            logging.info("Loaded model from local cache")
            model_path = cached_models[0]

        self.model = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(model_path)
        self.model.to(self._device)

    @property
    def suffix(self) -> str:
        return ".laion_clap.npy"

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = audio_tuple
        assert source_sr == 48000, "CLAP model only supports 48kHz audio"
        assert x.shape[0] == 1, "Audio must be mono"

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav")
        output_filename = remove_suffix(output_filename, ".mp3")
        output_filename += self.suffix

        # construct a batch from frames
        x = x.to(self._device)
        frames = []

        # the last frame is discarded else we cannot concatenate
        splits = x.split(self.num_samples_in_frame, dim=-1)[:-1]
        for frame in splits:
            frames.append(frame)

        if len(frames) == 0:
            logging.error(
                f"Skipping {key} because it is only {x.shape[1]} samples long, "
                f"and at least one frame of length {self.num_samples_in_frame} "
                "is required."
            )
            return []

        frames = torch.cat(frames, dim=0)

        # extract embeddings
        logging.info(f"Extracting CLAP embeddings for {key}")
        embeds = self.model.get_audio_embedding_from_data(x=frames, use_tensor=True)
        embeds = torch.transpose(embeds, 0, 1)  # -> [D, T]
        return [(output_filename, embeds.detach().cpu().numpy())]
