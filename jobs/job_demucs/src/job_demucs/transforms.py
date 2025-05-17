import apache_beam as beam
import apache_beam.io.fileio as beam_io
import apache_beam.io.filesystem as beam_fs
from apache_beam.io.filesystems import FileSystems
import av
import io
import logging
import numpy as np
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Tuple
import torchaudio
import torch
import traceback

from job_demucs.demucs import DemucsSeparator
from klay_beam.torch_transforms import convert_audio
from klay_beam.path import move, remove_suffix
from klay_beam.utils import get_device


VOCAL_CLASSIFIER_SUFFIX = ".voice_instrumental.npy"


class SeparateSources(beam.DoFn):

    def __init__(
        self,
        source_dir: str,
        target_dir: str,
        audio_suffix: str,
        target_audio_suffix: str,
        num_stems: int = 2,
        model_name: str = "hdemucs_mmi",
    ):
        """
        Arguments
        ---------
        source_dir
            The directory where the audio files are stored. Trim this from the
            source filename
        target_dir
            The directory where the output files should be stored.
        model_name
            Pretrained model name. The list of pre-trained models is:
                `mdx`: trained only on MusDB HQ, winning model on track A
                       at the MDX challenge.
                `mdx_extra`: trained with extra training data (including
                             MusDB test set). ranked 2nd on the track B of
                             the MDX challenge.
                `mdx_q`, `mdx_extra_q`: quantized version of the previous models.
                                        Smaller download and storage but quality
                                        can be slightly worse.
                `htdemucs_ft`: Fine tuned version of the default v4 model.
        """

        self.source_dir = source_dir
        self.target_dir = target_dir
        self.audio_suffix = audio_suffix
        self.target_audio_suffix = target_audio_suffix
        self.num_stems = num_stems
        self.model_name = model_name

    def setup(self):
        self.device = get_device()
        logging.info(f"Using device: {self.device}")

        # This will be executed only once when the pipeline starts.
        logging.info(f"Setting up DemucsSeparator: {self.model_name}")
        self.separator = DemucsSeparator(
            model_name=self.model_name,
            num_workers=1,
            device=self.device,
        )
        self.to_48k = torchaudio.transforms.Resample(44_100, 48_000)
        logging.info("DemucsSeparator setup")

    def _mux_stems(self, result_dict: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """
        Mux the instrumental stems together into a single tensor.

        Args:
            result_dict: Dictionary of 'vocals', 'bass', 'drums', and 'other' tensors.

        Returns:
            Dictionary with 'vocals' and 'instrumental' tensors.
        """
        assert self.num_stems == 2, "Muxing is only supported for 2 stems"
        return {
            "vocals": result_dict["vocals"],
            "instrumental": result_dict["bass"]
            + result_dict["drums"]
            + result_dict["other"],
        }

    def _create_output_triplets(
        self, input_filename: str, result_dict: dict[str, np.ndarray]
    ):
        """
        Create new filenames for the separated audio stems and return them as a list of triples.

        Args:
            input_filename: The original input filename.
            result_dict: Dictionary of separated audio stems with keys as stem names.

        Returns:
            List of tuples in the format: (filename, audio, sample_rate).
        """
        input_filestem = remove_suffix(input_filename, f".source{self.audio_suffix}")

        outputs = []

        for k, v in result_dict.items():
            new_suffix = f".{k}{self.target_audio_suffix}"
            new_filename = f"{input_filestem}{new_suffix}"

            outputs.append((new_filename, v, 44_100))

        return outputs

    def process(self, loaded_audio_tuple: Tuple[str, torch.Tensor, int]):
        start_time = time.time()
        key, audio_tensor, sr = loaded_audio_tuple

        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({key})"
        channel_count, sample_count = audio_tensor.shape
        durationSeconds = sample_count / sr

        if channel_count != 2:
            logging.error(
                f"Expected stereo audio. Found {channel_count} channels. ({key})"
            )
            return []

        input_filename = move(key, self.source_dir, self.target_dir)

        logging.info(f"Separating: {key}")
        try:
            result_dict = self.separator(audio_tensor)

            if self.num_stems == 2:
                result_dict = self._mux_stems(result_dict)

            triplets = self._create_output_triplets(input_filename, result_dict)

            elapsed_time = time.time() - start_time

            logging.info(
                "Separation complete! "
                f"Speed:{durationSeconds / elapsed_time:.3f}x realtime. "
                f"Audio Duration:{durationSeconds:.3f} seconds. "
                f"Processing time:{elapsed_time:.2f} seconds. "
                f"Key:{key}"
            )
            return triplets

        except Exception as e:
            logging.error(f"Exception while separating: {key}. Exception: {e}")
            return []


class FilterVocalAudio(beam.DoFn):
    def __init__(self, audio_suffix: str, only_if_vocals: bool = True):
        self.audio_suffix = audio_suffix
        self.only_if_vocals = only_if_vocals

    def process(self, file_metadata: beam_fs.FileMetadata):
        if not self.only_if_vocals:
            return [file_metadata]

        if not file_metadata.path.endswith(self.audio_suffix):
            raise ValueError(f"Invalid file suffix: {file_metadata.path}")

        vocal_classifier_path = file_metadata.path.replace(
            self.audio_suffix, VOCAL_CLASSIFIER_SUFFIX
        )

        # load the vocal classifier feats
        matches = beam.io.filesystems.FileSystems.match([vocal_classifier_path])[
            0
        ].metadata_list

        if not matches:
            logging.info(f"Skipping {file_metadata.path}, file not found.")
            return []

        readable_file = beam_io.ReadableFile(matches[0])

        with readable_file.open(mime_type="application/octet-stream") as file_like:
            feats = np.load(file_like)

        if IsVocalAudio.majority_vote(feats):
            return [file_metadata]
        else:
            logging.info(
                f"Skipping {file_metadata.path} because it does not contain vocals."
            )
            return []


class LoadNpy(beam.DoFn):
    """Load .npy files."""

    def process(self, readable_file: beam_io.ReadableFile):  # type: ignore
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, feats)` tuple where
            - `input_filename` is a string
            - `feats` is an np.ndarray
        """
        logging.info("Loading: {}".format(readable_file.metadata.path))
        with readable_file.open(mime_type="application/octet-stream") as file_like:
            feats = np.load(file_like)
        return [(readable_file.metadata.path, feats)]


class IsVocalAudio(beam.DoFn):
    """Compute whether the audio is vocal or not using majority voting."""

    # the following parameters indicate that we are considering an element as a vocal
    # if more than 10% of the frames have a greater than 90% probability of being vocal
    # we"re hard-coding these values for now, but they could be passed as arguments
    p_threshold: float = 0.9
    n_threshold: float = 0.1

    def __init__(self, p_threshold: float = 0.9, n_threshold: float = 0.1):
        self.p_threshold = p_threshold
        self.n_threshold = n_threshold

        logging.info(
            f"p_threshold: {self.p_threshold}; n_threshold: {self.n_threshold}"
        )

    @staticmethod
    def majority_vote(
        probs: np.ndarray, p_threshold: float = 0.9, n_threshold: int = 0.1
    ) -> bool:
        """Count the number of frames with a probability greater than `p_threshold`. If the majority
        of frames have a probability greater than `n_threshold`, it suggests that most frames likely
        contain a vocal.

        Args:
            probs: A 2D array of shape (2, T) where T is the number of frames and
                   the second row contains the probability of the frame containing vocals.
            p_threshold: The threshold for the probability of a frame containing vocals.
            n_threshold: The threshold for the number of frames containing vocals.
        Returns:
            True if the majority of frames contain vocals, False otherwise.
        """
        is_vocal_frames = probs[1] > p_threshold
        total_is_vocal = np.sum(is_vocal_frames) / probs.shape[1]
        return total_is_vocal >= n_threshold

    def process(self, element: Tuple[str, np.ndarray]):
        filepath, classifier_array = element
        is_vocal = self.majority_vote(
            classifier_array, self.p_threshold, self.n_threshold
        )

        if is_vocal:
            logging.info(f"Vocal: {filepath}")
        else:
            logging.info(f"No Vocal: {filepath}")

        yield filepath, is_vocal


class PrepareAudioFileKV(beam.DoFn):
    def process(self, audio_file: beam_io.ReadableFile):
        file_stem = audio_file.metadata.path.split("/")[-1].split(".")[0]
        yield (file_stem, audio_file)


class PrepareFeatureFileKV(beam.DoFn):
    def process(self, feature_file_tuple: Tuple[str, np.ndarray]):
        filepath, is_vocal = feature_file_tuple
        file_stem = filepath.split("/")[-1].split(".")[0]
        yield (file_stem, is_vocal)


class FilterAudioFilesByFeature(beam.DoFn):
    def process(self, element):
        key, grouped_data = element
        audio_files = grouped_data["audio_files"]
        is_vocal_results = grouped_data["is_vocal_results"]

        if len(audio_files) == 0:
            logging.info(f"Skipping {key} because missing audio")
            return []

        if len(is_vocal_results) == 0:
            logging.info(f"Skipping {key} because missing vocal features")
            return []

        # assuming a one-to-one mapping, so we directly access the first item
        if is_vocal_results and is_vocal_results[0]:
            yield audio_files[0]


class LoadWithTorchaudioDebug(beam.DoFn):
    def setup(self):
        torchaudio.set_audio_backend("soundfile")

    def process(self, readable_file: beam_io.ReadableFile):  # type: ignore
        path = Path(readable_file.metadata.path)

        # get the file extension without a period in a safe way
        ext_without_dot = path.suffix.lstrip(".")
        ext_without_dot = None if ext_without_dot == "" else ext_without_dot

        audio_tensor: torch.Tensor
        sr: int

        # try loading the audio file with torchaudio, but catch RuntimeError,
        # which are thrown when torchaudio can't load the file.
        logging.info("Loading: {}".format(path))
        try:
            with readable_file.open(mime_type="application/octet-stream") as file_like:
                audio_tensor, sr = torchaudio.load(file_like, format=ext_without_dot)
        except (RuntimeError, OSError) as e:
            logging.error(traceback.format_exc())
            return [beam.pvalue.TaggedOutput("failed", (str(path), e))]

        C, T = audio_tensor.shape
        duration_seconds = T / sr
        logging.info(f"Loaded {duration_seconds:.3f} second {C}-channel audio: {path}")

        return [
            (readable_file.metadata.path, audio_tensor, sr),
            beam.pvalue.TaggedOutput("duration_seconds", duration_seconds),
        ]


class CropAudioGTDuration(beam.DoFn):
    def __init__(self, max_duration: float):
        self.max_duration = max_duration

    def process(self, audio_tuple: Tuple[str, torch.Tensor, int]):
        key, audio, sr = audio_tuple

        if audio.shape[-1] / sr > self.max_duration:
            logging.info(
                f"File is greater than {self.max_duration}s long. Cropping: {key}"
            )
            return [(key, audio[..., : int(self.max_duration * sr)], sr)]

        return [audio_tuple]


def increment_numeric_index(path: str, increment: int = 4) -> str:
    """
    Finds the *first* occurrence of '.[digits].' in `path` and adds `increment`
    to that integer. For example, if increment=4:

        "gs://bucket/1000028872.0.source.stem.mp3"  -->  "gs://bucket/1000028872.4.source.stem.mp3"
        "gs://bucket/1000028872.9.source.stem.mp3"  -->  "gs://bucket/1000028872.13.source.stem.mp3"
        "gs://bucket/1000028872.source.stem.mp3"    -->  (unchanged if no numeric index)

    Only the *first* match is changed (count=1). If you need to alter *all* indices, remove `count=1`.
    """
    # We'll separate directory prefix from the actual filename so we don’t
    # accidentally mess with digits in the path structure. Then rejoin.
    prefix, slash, filename = path.rpartition("/")

    def replace_fn(match):
        old_index_str = match.group(1)  # the captured digits
        old_index = int(old_index_str)
        new_index = old_index + increment
        return f".{new_index}."

    new_filename = re.sub(r"\.(\d+)\.", replace_fn, filename, count=1)
    return prefix + slash + new_filename


class SkipCompleted(beam.DoFn):
    def __init__(
        self,
        old_suffix: str,
        new_suffix: str | list[str],
        source_dir: str | None = None,
        target_dir: str | None = None,
        check_timestamp: bool = False,
        overwrite: bool = False,
        shift_numeric_index: bool = True,
    ):
        if isinstance(new_suffix, str):
            new_suffix = [new_suffix]
        self._new_suffixes = new_suffix
        self._old_suffix = old_suffix

        assert (source_dir is None) == (
            target_dir is None
        ), "source_dir and target_dir must both be None or strings"

        self._source_dir = source_dir
        self._target_dir = target_dir
        self._check_timestamp = check_timestamp
        self._overwrite = overwrite
        self._shift_numeric_index = shift_numeric_index

    def process(self, source_metadata):
        """Given FileMetadata for a single file, decide if we skip it or process it."""
        if self._overwrite:
            yield source_metadata

        # Optionally remove the numeric index
        path_to_check = source_metadata.path
        if self._shift_numeric_index:
            path_to_check = increment_numeric_index(path_to_check, 4)

        # Remove the old suffix from the path
        stripped_path = remove_suffix(path_to_check, self._old_suffix)

        # Optionally swap directories
        if self._source_dir is not None:
            stripped_path = move(stripped_path, self._source_dir, self._target_dir)

        # For each possible "new" suffix, build a path to check if it already exists
        checks = [stripped_path + suffix for suffix in self._new_suffixes]
        limits = [1 for _ in checks]

        results = FileSystems.match(checks, limits=limits)
        # Should match len(checks)
        if not results:
            logging.warning("Unexpected empty results. This should never happen.")
            yield source_metadata
            return

        for result in results:
            num_matches = len(result.metadata_list)
            logging.info(f"Found {num_matches} of: {result.pattern}")

            if num_matches == 0:
                # If any requested path does not exist, we must process
                yield source_metadata
                return

            # If the file is found but is older than source, re-process
            for target_metadata in result.metadata_list:
                if (
                    target_metadata.last_updated_in_seconds
                    < source_metadata.last_updated_in_seconds
                ):
                    logging.info(
                        f"Do not skip! Target found ({target_metadata.path}) is older "
                        f"than source file ({source_metadata.path})."
                    )
                    yield source_metadata
                    return

        logging.info(f"Targets already exist. Skipping: {source_metadata.path}")


class LoadWebm(beam.DoFn):
    """DoFn that turns a .webm audio file into (path, np.ndarray, sample_rate)."""

    @staticmethod
    def _load_webm(buf: bytes) -> tuple[np.ndarray, int]:
        """
        Decode a WebM/Opus byte blob → float32 numpy array (samples, channels).

        args:
            buf : bytes  WebM/Opus byte blob

        returns:
            audio : np.ndarray  (samples, channels)
            sr    : int         sample-rate reported by the stream
        """
        container = av.open(io.BytesIO(buf))
        stream = next(s for s in container.streams if s.type == "audio")

        # Fallback if metadata is missing
        sr = None
        if hasattr(stream, "rate") and stream.rate is not None:
            sr = stream.rate

        frames = (f.to_ndarray() for f in container.decode(stream))
        audio = np.concatenate(list(frames), axis=1).T.astype(np.float32)
        return audio, sr

    def process(self, readable_file: beam_io.ReadableFile):  # type: ignore
        path = Path(readable_file.metadata.path)
        logging.info(f"Loading {path}")

        try:
            with readable_file.open(mime_type="application/octet-stream") as f:
                data = f.read()

            audio, sr = self._load_webm(data)

            if sr is None:
                logging.warning("Missing sample rate for %s", path)
                return
        except Exception as exc:
            logging.error(f"Error decoding {path} : {exc}")
            return

        audio = np.transpose(audio)
        duration = audio.shape[1] / sr
        logging.info(
            f"Loaded {duration:.4f}s, {audio.shape[0]}-channel audio  ↪  {path}"
        )
        yield readable_file.metadata.path, audio, sr


def numpy_to_vorbis(audio: np.ndarray, sr: int, q: float = 2.0) -> io.BytesIO:
    ch = audio.shape[0]
    raw = audio.T.astype(np.float32, copy=False).tobytes()
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "f32le",
        "-ar",
        str(sr),
        "-ac",
        str(ch),
        "-i",
        "pipe:0",
        "-vn",
        # vorbis is experimental so we need to use -strict -2
        "-c:a",
        "vorbis",
        "-strict",
        "-2",
        "-q:a",
        str(q),
        "-f",
        "ogg",
        "pipe:1",
    ]
    try:
        res = subprocess.run(
            cmd, input=raw, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr.decode(errors="ignore"))
        raise

    buf = io.BytesIO(res.stdout)
    buf.seek(0)
    return buf
