import apache_beam as beam
from apache_beam.coders import VarIntCoder
from apache_beam.transforms import userstate
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

from klay_data import pipeline


@dataclass(kw_only=True)
class LoadFeatureBeam:
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
    def __init__(self, min_duration: int, frame_rate: int) -> None:
        self.min_duration = min_duration
        self.frame_rate = frame_rate
        self.min_len = math.ceil(min_duration * self.frame_rate)
        self.uri_processing_errors = beam.metrics.Metrics.counter(
            ProcessURI.__name__, "uri_processing_errors"
        )
        self.uri_processed_successfully = beam.metrics.Metrics.counter(
            ProcessURI.__name__, "uri_processed_successfully"
        )
        self.p = pipeline.Pipeline(
            # Ensure audio is music.
            LoadFeatureBeam(
                input_key="audioset_yamnet.path", output_key="audioset_yamnet"
            ),
            pipeline.ValidateAudiosetYamnet(input_key="audioset_yamnet", feat_dim=521),
            pipeline.FilterIsMusic(input_key="audioset_yamnet"),
            # Load KlayNACVAE features.
            LoadFeatureBeam(input_key="klaynacvae.path", output_key="klaynacvae"),
            pipeline.ValidateKlayNACVAE(input_key="klaynacvae", feat_dim=128),
            pipeline.FilterLength(input_key="klaynacvae", min_len=self.min_len),
            # Load MTR++ features.
            LoadFeatureBeam(input_key="mtrpp.path", output_key="mtrpp"),
            pipeline.ValidateMTRPP(input_key="mtrpp", feat_dim=128),
            # Optionally load Whisper features.
            pipeline.Pipeline(
                LoadFeatureBeam(input_key="whisper.path", output_key="whisper"),
                is_optional=True,
                # pipeline.WrapFeature(input_key="whisper", dataclass=pipeline.WhisperLyricsInfo),
            ),
            pipeline.MaybeSetFeature(input_key="whisper", value={}),
            # pipeline.MaybeAddMissingWhisperLyrics(input_key="whisper"),
            # pipeline.FlattenFeature(input_key="whisper", output_prefix="whisper.")
        )

    def process(self, uri: str, *_):
        try:
            basename = os.path.basename(uri)

            # Construct the data dictionary with full paths
            data = {
                "whisper.path": f"{uri}/{basename}.vocals.whisper.json",
                "mtrpp.path": f"{uri}/{basename}.source.mtrpp.npy",
                "klaynacvae.path": f"{uri}/{basename}.source.klaynacvae-0.6.2.npy",
                "audioset_yamnet.path": f"{uri}/{basename}.source.audioset_yamnet.npy",
            }

            result = pipeline.run(self.p, data=data)
            self.uri_processed_successfully.inc()
            yield result
        except Exception as e:
            logging.error(f"Skipping {uri}: {e}")
            self.uri_processing_errors.inc()


class WriteMDS(beam.DoFn):
    def __init__(self, dest_dir: str) -> None:
        self.dest_dir = dest_dir
        self.columns = {
            "klaynacvae": "ndarray:float32",
            # "whisper.starts": "ndarray:float32",
            # "whisper.ends": "ndarray:float32",
            # "whisper.byt5_tokens": "ndarray:int32",
            # "whisper.byt5_embeds": "ndarray:float32",
            "whisper": "json",
            "mtrpp": "ndarray:float32",
        }
        self.writer = None
        self.worker_id = None

    def setup(self):
        # Create a unique subfolder for this worker
        self.worker_id = str(uuid.uuid4())
        worker_dir = os.path.join(self.dest_dir, f"worker_{self.worker_id}")
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


class _EnumerateDoFn(beam.DoFn):
    """
    Adds a monotonically-increasing integer to every element.
    Works in streaming or batch, on DirectRunner and Dataflow.
    """

    # One VALUE state cell per key, initialised to 0
    COUNTER = userstate.ReadModifyWriteStateSpec("c", VarIntCoder())

    def __init__(self, start: int = 0):
        self._start = start

    def process(
        self,
        kv,
        counter_state=beam.DoFn.StateParam(COUNTER),
    ):
        # kv comes from (None, element)
        _, element = kv

        current = counter_state.read() or self._start
        counter_state.write(current + 1)  # persist for the next element

        yield current, element


class Enumerate(beam.PTransform):
    """(e1, e2, …) → (0, e1), (1, e2), …"""

    def __init__(self, start: int = 0):
        self._start = start

    def expand(self, pcoll):
        return (
            pcoll
            | "KeyByNone" >> beam.Map(lambda x: (None, x))
            | "AttachIndex" >> beam.ParDo(_EnumerateDoFn(self._start))
        )
