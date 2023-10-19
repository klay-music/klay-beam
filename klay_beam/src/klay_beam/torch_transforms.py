from typing import Optional, Union, Tuple, List
import math
import logging
import numpy as np
import apache_beam as beam
import apache_beam.io.fileio as beam_io
import pathlib
import io
import scipy

from .path import remove_suffix
from .torch_utils import TORCH_AVAILABLE, TORCH_IMPORT_ERROR, ensure_torch_available


if TORCH_AVAILABLE:
    import torch
    import torchaudio
    from .extractors.spectral import ChromaExtractor


class LoadWithTorchaudio(beam.DoFn):
    """Use torchaudio to load audio files to tensors

    NOTES:

    - torchaudio depends on libavcodec, which can be installed with:
    `conda install 'ffmpeg<5'`. See:
    https://github.com/pytorch/audio/issues/2363#issuecomment-1179089175


    - Torchaudio supports loading in-memory (file-like) files since at least
    v0.9.0. See: https://pytorch.org/audio/0.9.0/backend.html#load


    Note that generally, custom functions have a few requirements that help them
    work well in on distributed runners. They are:
        - The function should be thread-compatible
        - The function should be serializable
        - Recommended: the function be idempotent

    For details about these requirements, see the Apache Beam documentation:
    https://beam.apache.org/documentation/programming-guide/#requirements-for-writing-user-code-for-beam-transforms
    """

    def setup(self):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        ensure_torch_available()
        pass

    def process(self, readable_file: beam_io.ReadableFile):
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, a, sr)` tuple where
            - `input_filename` is a string
            - `a` is a pytorch Tensor
            - `sr` is an int

        For a stereo audio file named '/path/to.some/file.key.mp3', return
        ```
        (
            '/path/to.some/file.key.mp3',
            tensor([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            44100
        )
        ```
        """

        # I could not find good documentation for beam ReadableFile, so I'm
        # putting the key information below.
        #
        # ReadableFile Properties
        # - readable_file.metadata.path
        # - readable_file.metadata.size_in_bytes
        # - readable_file.metadata.last_updated_in_seconds
        #
        # ReadableFile Methods
        # - readable_file.open(mime_type='text/plain', compression_type=None)
        # - readable_file.read(mime_type='application/octet-stream')
        # - readable_file.read_utf8()

        path = pathlib.Path(readable_file.metadata.path)

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
            # We don't want to log the stacktrace, but for debugging, here's how
            # we could access it:
            #
            # import traceback
            # tb_str = traceback.format_exception(
            #     etype=type(e), value=e, tb=e.__traceback__
            # )
            logging.warning(f"Error loading audio: {path}")
            return [beam.pvalue.TaggedOutput("failed", (str(path), e))]

        C, T = audio_tensor.shape
        duration_seconds = T / sr
        logging.info(f"Loaded {duration_seconds:.3f} second {C}-channel audio: {path}")

        return [
            (readable_file.metadata.path, audio_tensor, sr),
            beam.pvalue.TaggedOutput("duration_seconds", duration_seconds),
        ]


class ResampleTorchaudioTensor(beam.DoFn):
    """Resample an audio to a new sample rate. Accepts a `(key, a, sr)` tuple
    were:

    - `key` is a string
    - `a` is a 2D torch.Tensor or numpy.ndarray with audio in the last dimension
    - `sr` is an int

    The return value will also be a `(key, a, sr)` tuple, but 'a' will always be
    torch.Tensor instance.
    """

    MAX_AUDIO_CHANNELS = 128

    def __init__(
        self,
        target_sr: int,
        source_sr_hint: Optional[int] = None,
        output_numpy: bool = False,
    ):
        assert isinstance(
            target_sr, int
        ), f"target_sr must be an int (found {target_sr})"
        self._target_sr = target_sr
        self._source_sr_hint = source_sr_hint
        self._output_numpy = output_numpy
        self.resample = None

    def setup(self):
        ensure_torch_available()
        if self._source_sr_hint is not None:
            self.resample = torchaudio.transforms.Resample(
                self._source_sr_hint, self._target_sr
            )

    def process(self, audio_tuple: Tuple[str, Union["torch.Tensor", np.ndarray], int]):
        key, audio, source_sr = audio_tuple

        # check if audio_tensor is a numpy array or a torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        channels, _ = audio.shape
        if channels > self.MAX_AUDIO_CHANNELS:
            raise ValueError(
                f"audio_tensor ({key}) must have {self.MAX_AUDIO_CHANNELS} or "
                f"fewer channels (found {channels})"
            )

        resampled_audio: Optional[torch.Tensor] = None

        if source_sr == self._target_sr:
            resampled_audio = audio
            logging.info(
                f"Skipping resample because source was already {self._target_sr}: {key}"
            )
        elif self.resample is not None and source_sr == self._source_sr_hint:
            resampled_audio = self.resample(audio)
            logging.info(
                f"Resampled {source_sr} to {self._target_sr} (cached method): {key}"
            )
        else:
            resample = torchaudio.transforms.Resample(source_sr, self._target_sr)
            resampled_audio = resample(audio)
            logging.info(
                f"Resampled {source_sr} to {self._target_sr} (uncached method): {key}"
            )

        if self._output_numpy:
            resampled_audio = resampled_audio.numpy()

        return [(key, resampled_audio, self._target_sr)]


def convert_audio(wav: "torch.Tensor", sr: int, target_sr: int, target_channels: int):
    """Copied from encodec"""
    ensure_torch_available()
    if wav.ndim == 1:
        wav.unsqueeze_(0)
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


class ExtractChromaFeatures(beam.DoFn):
    """Extract features from an audio tensor. Accepts a `(key, a, sr)` tuple
    were:

    - `key` is a string
    - `a` is a 2D torch.Tensor or numpy.ndarray with audio in the last dimension
    - `sr` is an int

    The return value will also be a `(key, features)` tuple
    """

    def __init__(
        self,
        audio_sr: int,
        # The default values below are just copied from the ChromaExtractor
        # on August 10, 2023. If the defaults change in the future, should
        # we change them in both places? It would be nice to find a way not
        # to maintain two copies of the same default values.
        n_chroma: int = 12,
        n_fft: int = 2048,
        win_length: int = 2048,
        hop_length: Union[int, None] = None,
        norm: float = math.inf,
        device: Union["torch.device", str] = "cpu",
    ):
        self._audio_sr = audio_sr
        self._n_chroma = n_chroma
        self._n_fft = n_fft
        self._win_length = win_length
        self._hop_length = hop_length
        self._norm = norm
        self._device = device

    def setup(self):
        ensure_torch_available()
        self._chroma_model = ChromaExtractor(
            sample_rate=self._audio_sr,
            n_chroma=self._n_chroma,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            norm=self._norm,
            device=self._device,
        )

    def process(self, element: Tuple[str, "torch.Tensor", int]):
        key, audio, sr = element

        try:
            # Ensure correct sample rate, and convert to mono
            audio = convert_audio(audio, sr, self._audio_sr, 1)

            features = self._chroma_model(audio)
            output_path = remove_suffix(key, ".wav") + self._chroma_model.feat_suffix

            logging.info(
                f"Extracted chroma ({features.shape}) from audio ({audio.shape}): {output_path}"
            )

            return [(output_path, features)]

        except Exception as e:
            logging.error(f"Failed to extract chroma features for {key}: {e}")
            return []


def tensor_to_bytes(
    audio_tuple: Tuple[str, Union["torch.Tensor", np.ndarray], int]
) -> List[Tuple[str, bytes, int]]:
    fname, audio, sr = audio_tuple
    if isinstance(audio, torch.Tensor):
        audio = audio.detach().cpu().numpy()

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    buf.seek(0)
    wav_data = buf.read()

    return [(fname, wav_data, sr)]
