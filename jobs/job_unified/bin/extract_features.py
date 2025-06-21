from functools import lru_cache
from dataclasses import dataclass
import math
import logging
from typing import Any, Protocol
import torch
from torch.utils.data import Dataset, DataLoader
from klay_data.storage_client import get_storage_client, StorageClient
import numpy as np
from io import BytesIO

from klay_codecs.nac import KlayNACVAE
from klay_data import pipeline


class Processor(Protocol):
    def to(self, device: torch.device): ...
    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]: ...


def make_frames(
    audio: torch.tensor, window_length: int, hop_length: int
) -> list[torch.tensor]:
    """Slice audio into overlapping windows.

    Args:
        audio: Tensor of shape (D, T) to be sliced into windows.
        window_length: Length of each window in samples.
        hop_length: Hop size in samples.
    """
    _, T = audio.shape

    if T < window_length:
        return [audio]

    starts = torch.arange(0, T, hop_length).tolist()
    windows = [audio[:, s : s + window_length] for s in starts]

    overlap = window_length - hop_length
    if windows[-1].shape[-1] < overlap:
        # If the last window is shorter than the overlap, drop it
        windows = windows[:-1]
    return windows


def make_fade_curves(
    overlap: int, device: torch.device
) -> tuple[torch.tensor, torch.tensor]:
    """Return linear fade-in and fade-out curves."""
    fade_in = torch.linspace(0.0, 1.0, overlap, device=device)
    fade_out = torch.flip(fade_in, dims=[0])
    return fade_in, fade_out


def make_envelope(
    index: int, total: int, length: int, overlap: int, device: torch.device
) -> torch.tensor:
    """Create asymmetric cross-fade window for overlap-add."""
    fade_in, fade_out = make_fade_curves(overlap, device)
    w = torch.ones(length, device=device)

    if index > 0:
        # not first → ramp up
        w[:overlap] = fade_in
    if index < total - 1:
        # not last → ramp down
        w[-overlap:] = fade_out

    return w.unsqueeze(0)


def overlap_add(
    tensors: list[torch.tensor], hop_length: int, total_length: int
) -> torch.tensor:
    """Linear overlap add along the time axis with asymmetric ramps."""
    if len(tensors) == 1:
        return tensors[0]

    D, window_length = tensors[0].shape
    overlap = window_length - hop_length
    device = tensors[0].device

    # Pre-allocate output tensor on same device as input
    out = torch.zeros(D, total_length, device=device)

    # Pre-compute fade curves for efficiency
    fade_in, fade_out = make_fade_curves(overlap, device)

    # Process frames in place
    for idx, frame in enumerate(tensors):
        if frame.shape[-1] != window_length:
            envelope_length = frame.shape[-1]
        else:
            envelope_length = window_length

        # Create envelope (reuse fade curves)
        w = torch.ones(envelope_length, device=device)
        if idx > 0:
            w[:overlap] = fade_in
        if idx < len(tensors) - 1:
            w[-overlap:] = fade_out
        w = w.unsqueeze(0)  # Add channel dimension

        # Apply envelope and add to output
        start = idx * hop_length
        end = start + window_length
        out[:, start:end] += frame * w

    return out


def secs_to_samples(seconds: float, rate: int) -> int:
    """Convert seconds to number of samples."""
    return math.ceil(seconds * rate)


@dataclass(kw_only=True)
class KlayNACConfig:
    window_duration: float = 195.0
    hop_duration: float = 190.0

    def setup(self):
        return KlayNACProcessor(self)


class KlayNACProcessor(Processor):
    def __init__(self, config: KlayNACConfig):
        self.config = config
        self.nac = KlayNACVAE(dummy_mode=False)

    @property
    def audio_window_length(self) -> int:
        return secs_to_samples(self.config.window_duration, self.nac.config.sample_rate)

    @property
    def audio_hop_length(self) -> int:
        return secs_to_samples(self.config.hop_duration, self.nac.config.sample_rate)

    @property
    def embed_window_length(self) -> int:
        return secs_to_samples(self.config.window_duration, self.nac.config.frame_rate)

    @property
    def embed_hop_length(self) -> int:
        return secs_to_samples(self.config.hop_duration, self.nac.config.frame_rate)

    def __call__(self, audio: torch.tensor) -> torch.tensor:
        """Apply the KlayNAC model to process audio frames."""

        # Create frames
        audio_frames = make_frames(
            audio, self.audio_window_length, self.audio_hop_length
        )
        embed_frames = []

        # Process frames
        with torch.no_grad():
            for audio_frame in audio_frames:
                embeds, _ = self.nac.audio_to_embeds(audio_frame.unsqueeze(0))
                embed_frames.append(embeds[0])

            # Overlap-add the frames
            sr = self.nac.config.sample_rate
            output_array = overlap_add(
                embed_frames,
                hop_length=self.embed_hop_length,
                total_length=int((audio.shape[-1] / sr) * self.nac.config.frame_rate),
            )

            if torch.isnan(output_array).any():
                raise ValueError("NaN values detected in output array")

        return output_array


@lru_cache(maxsize=1)
def global_storage_client() -> StorageClient:
    return get_storage_client()


class PipelineDataset(Dataset):
    def __init__(
        self,
        manifest: list[str],
        pipeline: pipeline.Pipeline,
        bucket_name: str,
    ):
        self.manifest = manifest
        self.pipeline = pipeline
        self.bucket_name = bucket_name

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, idx: int):
        filename = self.manifest[idx]
        try:
            return pipeline.run(
                self.pipeline,
                dict(filename=filename),
                storage_client=global_storage_client(),
                bucket_name=self.bucket_name,
            )
        except Exception as e:
            print(f"Data pipeline error for {filename}: {e}")
            return {}


@dataclass(kw_only=True)
class ApplyModel:
    input_key: str
    model_name: str

    def __call__(
        self, data: dict[str, Any], device: torch.device, **kwargs
    ) -> dict[str, Any]:
        model = kwargs.get(self.model_name)
        if model is None:
            raise ValueError(f"Model {self.model_name} not found")

        logging.info(f"Running {self.model_name} model on {device}")

        model.to(device)
        output = model(data[self.input_key])
        model.to(torch.device("cpu"))

        return {self.model_name: output}


def append_suffix(filename: str, suffix: str) -> str:
    return filename.rsplit(".", 1)[0] + suffix


@dataclass(kw_only=True)
class SaveFeature:
    input_key: str
    suffix: str
    filename_key: str = "filename"

    def __call__(
        self,
        data: dict[str, Any],
        storage_client: StorageClient = None,
        bucket_name: str = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Save a tensor to disk with the specified suffix."""
        if self.input_key not in data:
            raise ValueError(f"Input key '{self.input_key}' not found in data")

        tensor = data[self.input_key]
        output_filename = append_suffix(data[self.filename_key], self.suffix)

        # Convert tensor to numpy and save
        if isinstance(tensor, torch.Tensor):
            numpy_array = tensor.detach().cpu().numpy()
        else:
            numpy_array = tensor

        try:
            if storage_client and bucket_name:
                buffer = BytesIO()
                np.save(buffer, numpy_array)
                buffer.seek(0)
                storage_client.save_blob(bucket_name, output_filename, buffer)
                logging.info(f"Saved feature {self.input_key} to {bucket_name}/{output_filename}")
            else:
                np.save(output_filename, numpy_array)
                logging.info(f"Saved feature {self.input_key} to {output_filename}")
        except Exception as e:
            logging.error(f"Failed to save feature {self.input_key} to {output_filename}: {e}")
            raise

        return data  # Return data unchanged


@dataclass(kw_only=True)
class FilterIncompleteFeature:
    """Filter out features that already exist on disk."""

    suffix: str
    filename_key: str = "filename"

    def __call__(
        self,
        data: dict[str, Any],
        storage_client: StorageClient,
        bucket_name: str,
        **kwargs,
    ) -> dict[str, Any]:
        output_filename = append_suffix(data[self.filename_key], self.suffix)
        if storage_client.exists(bucket_name, output_filename):
            raise pipeline.SkipPipeline(f"Skip complete feature: {output_filename}")
        return {}


def go(manifest: list[str], bucket_name: str, dataset: str):
    # Set up data loading pipeline.
    loading_pipeline = pipeline.Pipeline(
        pipeline.LoadFeature(
            input_key="filename",
            output_key="audio",
        ),
        # TODO: Resample to 16 kHz for essentia
        # TODO: Resample to 41 kHz for dmucs
        pipeline.FlattenFeature(
            input_key="audio",
            output_prefix="input.",
        ),
        pipeline.KeepKeys("filename", "input.audio"),  # , "input.sample_rate"),
    )

    dataset = PipelineDataset(
        manifest,
        loading_pipeline,
        bucket_name=bucket_name,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        prefetch_factor=1,
    )

    # Set up models.
    model_configs = {
        "klaynac": KlayNACConfig(),
        # "mtrpp":
        # "demucs":
        # "byt5":
    }

    models = {}
    for model_name, model_config in model_configs.items():
        logging.info(f"Setting up {model_name} model...")
        models[model_name] = model_config.setup()

    # Set up processing pipeline.
    pipes = {}
    pipes["klaynac"] = pipeline.Pipeline(
        FilterIncompleteFeature(suffix=".klaynacvae-0.6.2.npy"),
        ApplyModel(input_key="input.audio", model_name="klaynac"),
        SaveFeature(input_key="klaynac", suffix=".klaynacvae-0.6.2.npy"),
    )

    # TODO: Run essentia pipeline in loading pipeline.
    # pipes["essentia"] = pipeline.Pipeline(
    # TODO: Extract audioset_yamnet features.
    # )

    # TODO: Run demucs pipeline in processing pipeline.
    # pipes["demucs"] = pipeline.Pipeline(
    #     pipeline.FilterHasVocals(),
    #     ResampleAudio(input_key="input.audio", sample_rate=41_000),
    #     ApplyModel(input_key="input.audio", model_name="demucs"),
    #     ResampleAudio(input_key="demucs.vocals", sample_rate=48_000),
    #     ResampleAudio(input_key="demucs.instrumental", sample_rate=48_000),
    #     SaveFeature(input_key="demucs.vocals", suffix=".vocals.npy"),
    #     SaveFeature(input_key="demucs.instrumental", suffix=".instrumental.npy"),
    #     ApplyModel(input_key="demucs.vocals", model_name="klaynac", output_key=""),
    # )

    # TODO: Run mtrpp pipeline in processing pipeline.
    # pipes["mtrpp"] = pipeline.Pipeline(
    #     ApplyModel(input_key="input.audio", model_name="mtrpp"),
    #     SaveFeature(input_key="mtrpp", suffix=".mtrpp.npy"),
    # )

    # TODO: Run whisper pipeline in processing pipeline.
    # pipes["whisper"] = pipeline.Pipeline(
    #     ApplyModel(input_key="input.audio", model_name="whisper"),
    #     SaveFeature(input_key="whisper", suffix=".whisper.npy"),
    #     ApplyModel(input_key="whisper", model_name="byt5"),
    #     SaveFeature(input_key="byt5", suffix=".byt5.npy"),
    # )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_failures = 0

    # TODO: Select pipelines to use.
    processing_pipeline = pipeline.Pipeline(*pipes.values())

    # Main loop, processing one file at a time.
    for count, data in enumerate(dataloader, 1):
        if not data:
            num_failures += 1
            logging.error(f"Failed {num_failures} out of {count} files seen so far")
            continue

        # Assume batch size of 1 and transfer tensors to device.
        for k, v in data.items():
            data[k] = v[0]
            if isinstance(v, torch.Tensor):
                data[k].to(device)

        filename = data["filename"]
        logging.info(f"[{count}/{len(manifest)}] Processing {filename}")

        try:
            pipeline.run(
                processing_pipeline,
                data,
                device=device,
                storage_client=global_storage_client(),
                bucket_name=bucket_name,
                **models,
            )
        except Exception as e:
            num_failures += 1
            logging.error(f"Processing pipeline error for {filename}: {e}")
            logging.error(f"Failed {num_failures} out of {count} files")

    # Log summary stats.
    num_processed = len(manifest) - num_failures
    logging.info(f"Successfully processed {num_processed} files out of {len(manifest)}")
    logging.info(f"Failed to process {num_failures} files")


if __name__ == "__main__":
    # TODO: configure log level with a flag
    logging.basicConfig(
        level=getattr(logging, "INFO"),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # TODO: read bucket_name and dataset from flags
    bucket_name = "klay-datasets-test"
    dataset = "test-shard"
    # TODO: read manifest from flag
    manifest = [
        f"{dataset}/-_FQ-xzvwws/-_FQ-xzvwws.source.ogg",
        # f"{dataset}/-0iGuO626zs/-0iGuO626zs.source.ogg",
    ]
    go(manifest, bucket_name, dataset)
