import apache_beam as beam
import apache_beam.io.fileio as beam_io
import av
import io
import logging
import numpy as np
from pathlib import Path
import subprocess
import sys

from klay_beam.path import remove_suffix
from klay_beam.transforms import (
    numpy_to_wav,
    numpy_to_file,
)


FFMPEG_BIN = "ffmpeg"


def numpy_to_vorbis(audio: np.ndarray, sr: int, q: float = 2.0) -> io.BytesIO:
    ch = audio.shape[0]
    raw = audio.T.astype(np.float32, copy=False).tobytes()
    cmd = [
        FFMPEG_BIN,
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


def numpy_to_mp3(audio: np.ndarray, sr: int, kbps: int = 192) -> io.BytesIO:
    """Encode (channels, samples) float32 ndarray → MP3 via FFmpeg/libmp3lame."""
    ch = audio.shape[0]
    raw = audio.T.astype(np.float32, copy=False).tobytes()
    cmd = [
        FFMPEG_BIN,
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
        "-c:a",
        "libmp3lame",
        "-b:a",
        f"{kbps}k",
        "-f",
        "mp3",
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


def crop_or_skip_audio(audio: np.ndarray, crop_length: int):
    """
    Take a random crop of target length of the audio.
    If the audio is shorter than the target length, return None.

    Args:
        audio (np.ndarray): The audio to crop, shape (channels, num_samples).
        crop_length (int): The target length of the crop.
    """
    num_samples = audio.shape[1]
    if num_samples <= crop_length:
        return None

    crop_start = np.random.randint(0, num_samples - crop_length)
    crop_end = crop_start + crop_length
    return audio[:, crop_start:crop_end]


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


class TranscodeFn(beam.DoFn):
    def __init__(
        self,
        crop_duration: float | None,
        target_sample_rate: int,
        audio_suffix: str,
        target_audio_suffix: str,
    ):
        self.crop_duration = crop_duration
        self.target_sample_rate = target_sample_rate
        self.audio_suffix = audio_suffix
        self.target_audio_suffix = target_audio_suffix

    @property
    def target_suffix(self):
        if self.crop_duration is None:
            return self.target_audio_suffix
        else:
            return f"-{self.crop_duration}s{self.target_audio_suffix}"

    def process(self, element):
        """
        Transcode the audio file to numpy format.

        Args:
            element (tuple): A tuple containing the key, audio data, and sample rate.

        Returns:
            tuple: (New key, transcoded audio data)
        """
        key, audio, sr = element

        assert (
            sr == self.target_sample_rate
        ), "Only 48kHz audio is supported for numpy files"

        if self.crop_duration is not None:
            num_samples = int(self.crop_duration * sr)
            audio = crop_or_skip_audio(audio, num_samples)

            if audio is None:
                return

        new_key = remove_suffix(key, self.audio_suffix) + self.target_suffix

        if not isinstance(audio, np.ndarray):
            audio = audio.numpy()

        if self.target_audio_suffix == ".npy":
            yield new_key, numpy_to_file(audio)
        elif self.target_audio_suffix == ".mp3":
            yield new_key, numpy_to_mp3(audio, sr)
        elif self.target_audio_suffix == ".wav":
            yield new_key, numpy_to_wav(audio, sr)
        elif self.target_audio_suffix == ".ogg":
            yield new_key, numpy_to_vorbis(audio, sr)
