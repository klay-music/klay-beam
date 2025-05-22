import apache_beam as beam
from dataclasses import dataclass
from typing import Any
import numpy as np
import json
import io
import math
import uuid
from streaming import MDSWriter
import logging
import os

from klay_data.pipeline import (
    Pipeline,
    ValidateAudiosetYamnet,
    FilterIsMusic,
    ValidateKlayNACVAE,
    FilterLength,
    ValidateMTRPP,
    MaybeSetFeature,
    WrapFeature,
    FlattenFeature,
    WhisperLyricsInfo,
    run as run_pipeline,
)


# Define all the suffixes for the features that may be included in the input.
FEATURE_SUFFIX = {
    "audioset_yamnet": ".source.audioset_yamnet.npy",
    "klaynacvae": ".source.klaynacvae-0.6.2.npy",
    "mtrpp": ".source.mtrpp.npy",
    "whisper_byt5": ".vocals.whisper-byt5.npz",
    "whisper": ".vocals.whisper.json",
}


# Define all the columns that may be included in the MDS output. Note that some features have
# multiple columns, which is indicated by the "." in the key.
FEATURE_MDS_TYPE = {
    "klaynacvae": "ndarray:float32",
    "mtrpp": "ndarray:float32",
    "whisper_byt5.byt5_embeds": "ndarray:float32",
    "whisper_byt5.byt5_tokens": "ndarray:int32",
    "whisper_byt5.ends": "ndarray:float32",
    "whisper_byt5.starts": "ndarray:float32",
    "whisper": "json",
}


# NOTE: This class is defined here because we don't want the Beam dependency inside klay-data.
@dataclass(kw_only=True)
class LoadFeatureBeam:
    """Load a feature in a Beam pipeline."""

    input_key: str
    output_key: str

    def __call__(self, data: dict[str, Any], **kwargs) -> dict[str, Any]:
        path = data[self.input_key]

        # Check if the file exists before attempting to open. This is because Beam lazily opens
        # files and we want a clear error if the file doesn't exist.
        if not beam.io.filesystems.FileSystems.exists(path):
            raise FileNotFoundError(f"File not found: {path}")

        _, extension = os.path.splitext(path)

        match extension:
            case ".npy":
                with beam.io.filesystems.FileSystems.open(
                    path, mime_type="application/octet-stream"
                ) as f:
                    content = np.load(io.BytesIO(f.read()))
            case ".json":
                with beam.io.filesystems.FileSystems.open(
                    path, mime_type="text/plain"
                ) as f:
                    content = json.load(f)
            case _:
                raise ValueError(
                    f"Unsupported file extension '{extension}' for path {path}"
                )

        return {self.output_key: content}


class ProcessURI(beam.DoFn):
    def __init__(
        self, min_duration: int, frame_rate: int, suffixes: dict[str, str]
    ) -> None:
        self.min_duration = min_duration
        self.frame_rate = frame_rate
        self.min_len = math.ceil(min_duration * self.frame_rate)
        self.suffixes = suffixes
        self.uri_processing_errors = beam.metrics.Metrics.counter(
            ProcessURI.__name__, "uri_processing_errors"
        )
        self.uri_processed_successfully = beam.metrics.Metrics.counter(
            ProcessURI.__name__, "uri_processed_successfully"
        )

        stages = {}

        # Ensure audio is music.
        stages["audioset_yamnet"] = Pipeline(
            LoadFeatureBeam(
                input_key="audioset_yamnet.path", output_key="audioset_yamnet"
            ),
            ValidateAudiosetYamnet(input_key="audioset_yamnet", feat_dim=521),
            FilterIsMusic(input_key="audioset_yamnet"),
        )

        # Load KlayNACVAE features.
        stages["klaynacvae"] = Pipeline(
            LoadFeatureBeam(input_key="klaynacvae.path", output_key="klaynacvae"),
            ValidateKlayNACVAE(input_key="klaynacvae", feat_dim=128),
            FilterLength(input_key="klaynacvae", min_len=self.min_len),
        )

        # Load MTR++ features.
        stages["mtrpp"] = Pipeline(
            LoadFeatureBeam(input_key="mtrpp.path", output_key="mtrpp"),
            ValidateMTRPP(input_key="mtrpp", feat_dim=128),
        )

        # Optionally load Whisper features.
        stages["whisper"] = Pipeline(
            Pipeline(
                LoadFeatureBeam(input_key="whisper.path", output_key="whisper"),
                # TODO: Validate Whisper features.
                is_optional=True,
            ),
            MaybeSetFeature(input_key="whisper", value={}),
        )

        # Optionally load Whisper ByT5 features.
        stages["whisper_byt5"] = Pipeline(
            Pipeline(
                LoadFeatureBeam(
                    input_key="whisper_byt5.path", output_key="whisper_byt5"
                ),
                # TODO: Validate Whisper ByT5 features.
                WrapFeature(input_key="whisper_byt5", dataclass=WhisperLyricsInfo),
                is_optional=True,
            ),
            MaybeSetFeature(
                input_key="whisper_byt5",
                value=WhisperLyricsInfo(
                    starts=np.zeros((1,), dtype=np.float32),
                    ends=np.zeros((1,), dtype=np.float32),
                    byt5_tokens=np.full((1, 1), -1, dtype=np.int32),
                    byt5_embeds=np.zeros((1, 1), dtype=np.float32),
                ),
            ),
            FlattenFeature(input_key="whisper_byt5", output_prefix="whisper_byt5."),
        )

        # NOTE! This code is carefully written so that the order of stages in `self.pipeline` is the
        # same as the order defined in the code above. In particular, `include_stage` is the same
        # order as `stages``. This is essential because, e.g. music filtering should happen first.
        include_stage = {k: k in self.suffixes for k in stages}
        self.pipeline = Pipeline(*[v for k, v in stages.items() if include_stage[k]])

    def process(self, uri: str, *_):
        try:
            # Construct paths to features. These are loaded by LoadFeatureBeam stages.
            basename = os.path.basename(uri)
            data = {
                k + ".path": f"{uri}/{basename}{v}" for k, v in self.suffixes.items()
            }

            result = run_pipeline(self.pipeline, data=data)
            self.uri_processed_successfully.inc()
            yield result
        except Exception as e:
            logging.error(f"Skipping {uri}: {e}")
            self.uri_processing_errors.inc()


class WriteMDS(beam.DoFn):
    def __init__(self, dest_dir: str, features: list[str]) -> None:
        self.dest_dir = dest_dir
        # The MDS types have keys that are either feature names or feature names with a "." and a
        # suffix, so make sure to include both in the MDS output.
        self.columns = {
            k: v for k, v in FEATURE_MDS_TYPE.items() if k.split(".")[0] in features
        }
        self.writer = None
        self.worker_id = None

    def setup(self):
        # Create a unique subfolder for this worker
        self.worker_id = str(uuid.uuid4())
        worker_dir = os.path.join(self.dest_dir, self.worker_id)
        os.makedirs(worker_dir, exist_ok=True)

        self.writer = MDSWriter(
            out=worker_dir,
            columns=self.columns,
            compression="zstd",
            hashes=["xxh3_64"],
        )

    def process(self, element, *_):
        self.writer.write(element)
        yield None

    def teardown(self):
        self.writer.finish()
