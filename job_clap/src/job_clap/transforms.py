from audiotools import AudioSignal
from dac.utils import load_model
from dac.model import DAC
from dac.utils.encode import process as encode
from encodec import EncodecModel
from encodec.compress import compress_to_file as create_ecdc
from encodec.compress import decompress as decompress_ecdc
import logging
import torch
import io
from typing import Tuple, Optional

import apache_beam as beam

from klay_beam.transforms import convert_audio
from klay_beam.path import remove_suffix
from klay_beam.utils import get_device


REPO_ID = "lukewys/laion_clap"
CACHE_DIR = Path.home() / ".cache/huggingface/hub/models--lukewys--laion_clap/snapshots"
FILENAME = "music_audioset_epoch_15_esc_90.14.pt"


class ExtractCLAP(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    sample_rate = 48000

    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def setup(self):
        cached_models = list(CACHE_DIR.glob(FILENAME))
        if not cached_models:
            model_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME)
        else:
            model_path = cached_models[0]

        self.model = laion_clap.CLAP_module(enable_fusion=False, amodel="HTSAT-base")
        self.model.load_ckpt(model_path)
        self.model.to(self._device)

    @property
    def suffix(self) -> str:
        return ".laion_clap.npy"

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element
        assert source_sr == 48000, "CLAP model only supports 48kHz audio"
        assert x.shape[0] in [1, 2], "Audio is not formatted correctly, channel must be first"

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav")
        output_filename = remove_suffix(output_filename, ".mp3")
        output_filename += self.suffix

        # extract embeddings
        x = x.to(self._device)
        embeds = self.model.get_audio_embedding_from_data(x=x, use_tensor=True)
        return [(output_filename, embeds)]
