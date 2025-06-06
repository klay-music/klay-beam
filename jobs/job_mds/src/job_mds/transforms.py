import apache_beam as beam
from dataclasses import dataclass
import numpy as np
import io
import json
import logging
import math
import os
import uuid
from streaming import MDSWriter
from time import sleep
from typing import Any

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
    "audioset_yamnet": ".0.source.stem.audioset_yamnet.npy",
    "klaynacvae": ".0.source.stem.klaynacvae-0.5.0.npy",
    "mtrpp": ".0.source.stem.mtrpp.npy",
    "whisper_byt5": ".4.vocals.stem.whisper-byt5.npz",
    "whisper": ".4.vocals.stem.whisper.json",
    "vocal_stem": ".4.vocals.stem.mp3",
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
    "vocal_stem": "bytes",
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
            case ".mp3":
                with beam.io.filesystems.FileSystems.open(
                    path, mime_type="audio/mpeg"
                ) as f:
                    content = f.read()
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

        # Optionally load vocal stem.
        stages["vocal_stem"] = Pipeline(
            Pipeline(
                LoadFeatureBeam(input_key="vocal_stem.path", output_key="vocal_stem"),
                is_optional=True,
            ),
            MaybeSetFeature(input_key="vocal_stem", value=b""),
        )

        # NOTE! This code is carefully written so that the order of stages in `self.pipeline` is the
        # same as the order defined in the code above. In particular, `include_stage` is the same
        # order as `stages``. This is essential because, e.g. music filtering should happen first.
        include_stage = {k: k in self.suffixes for k in stages}
        self.pipeline = Pipeline(*[v for k, v in stages.items() if include_stage[k]])

    def process(self, uri: str, *_):
        """Process a URI.

        Args:
            uri: The URI of the audio file to process. This does not contain the file name, only
                the name of the folder, e.g. gs://klay-datasets-test/shard-0000/gxLdBEQBAOU.

        Returns:
            A dictionary containing the processed features.
        """
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
            return


class WriteMDS(beam.DoFn):
    def __init__(self, dest_dir: str, features: list[str]) -> None:
        self.dest_dir = dest_dir
        # The MDS types have keys that are either feature names or feature names with a "." and a
        # suffix, so make sure to include both in the MDS output.
        self.columns = {
            k: v for k, v in FEATURE_MDS_TYPE.items() if k.split(".")[0] in features
        }
        self.worker_id = None
        self._closed = False

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
            size_limit=1 << 28,  # 256 MB
            max_workers=1,
        )

    def process(self, element, *_):
        # logging.info(f"Writing MDS for worker {self.worker_id}, element: {element}")
        self.writer.write(element)
        yield None

    def finish_bundle(self):
        if not self._closed:
            logging.info(f"Finishing MDSWrite bundle for worker {self.worker_id}.")
            retry_count = 0
            while True:
                if retry_count > 5:
                    logging.error(
                        f"Failed to finish MDS for worker {self.worker_id} after 5 attempts."
                    )
                    break
                try:
                    self.writer.finish()
                except FileNotFoundError:
                    logging.warning(
                        f"Attempt {retry_count}: File write incomplete for worker {self.worker_id},"
                        f" retrying in {retry_count * 10}s..."
                    )
                    retry_count += 1
                    sleep(retry_count * 10)
                except Exception as e:
                    logging.error(
                        f"Failed to finish MDS for worker {self.worker_id}: {e}"
                    )
                    break

            self._closed = True
