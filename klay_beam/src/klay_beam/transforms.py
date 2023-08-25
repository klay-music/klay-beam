import pathlib
import io
import torch
from typing import Optional, Type, Union, Tuple, List
from packaging import version as packaging_version
import apache_beam as beam
from apache_beam.io import filesystems
import torchaudio
import pydub
import soundfile as sf
import numpy as np
import logging

from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

from klay_data.transform import convert_audio
from .extractors.spectral import ChromaExtractor
from .path import move, remove_suffix


def numpy_to_pydub_audio_segment(
    audio_data: np.ndarray, sr: int, bit_depth=16
) -> pydub.AudioSegment:
    assert audio_data.ndim == 2, "audio_data must be 2 dimensional"
    channels, _ = audio_data.shape

    assert channels in [
        1,
        2,
    ], f"PyDub only supports mono and stereo audio (found {channels} channels)"

    # PuDub only supports 8 and 16 bit depth.
    assert bit_depth in [8, 16], f"bit_depth must be 8 or 16 (found {bit_depth}))"

    audio_data = audio_data.T  # pydub expects interleaved audio
    audio_data = audio_data * (2 ** (bit_depth - 1) - 1)  # scale to bit_depth.

    np_type: Optional[Type[np.signedinteger]] = None
    sample_width: Optional[int] = None
    if bit_depth == 8:
        np_type = np.int8
        sample_width = 1
    elif bit_depth == 16:
        np_type = np.int16
        sample_width = 2
    else:
        assert False, f"bit_depth must be 8, 16 (found {bit_depth}))"

    audio_data = audio_data.astype(np_type)
    raw_audio_buffer = io.BytesIO(audio_data.tobytes())

    audio_segment = pydub.AudioSegment.from_raw(
        raw_audio_buffer, frame_rate=sr, channels=channels, sample_width=sample_width
    )

    return audio_segment


def numpy_to_mp3(
    audio_data: np.ndarray,
    sr: int,
    bitrate="256k",
    intermediary_bit_depth=16,
):
    """Convert the audio data to an in-memory mp3 encoded file-like object."""
    channels, samples = audio_data.shape
    assert (
        samples > channels
    ), "incoming audio data must not be interleaved (last dimension must be audio)"

    logging.info(
        "Creating mp3: length: {:.3f} seconds. Channels: {}".format(
            samples / sr, channels
        )
    )

    # This needs to write a file-like object in memory. Loading file-like objects
    # is now supported by torchaudio (both sox_io and soundfile backends), but
    # torchaudio.save can only write to disk. As a result, we use pydub, which
    # invokes ffmpeg to write to memory.

    # For pytorch support see: https://pytorch.org/audio/stable/backend.html
    # For pydub docs see: https://github.com/jiaaro/pydub
    audio_segment = numpy_to_pydub_audio_segment(
        audio_data, sr, bit_depth=intermediary_bit_depth
    )

    # create an in-memory file-like object to write to
    in_memory_file_buffer = io.BytesIO()

    # Encode the pydub.AudioSegment as an mp3 (in-memory).
    #
    # ```
    # # Figure out which encoders are available (libmp3lame):
    # ffmpeg -encoders | grep mp3
    #
    # # Identify supported bit-rates
    # ffmpeg -h encoder=libmp3lame
    # ```
    audio_segment.export(in_memory_file_buffer, format="mp3", bitrate=bitrate)
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


# `pip install soundfile` has buggy vorbis support. soundfile can only
# reliably write ogg files when the underlying libsndfile is 1.2.0 or greater
# You can check this with `soundfile.__libsndfile_version__`. For details,
# see: https://github.com/bastibe/python-soundfile/issues/130
sf_current_version = packaging_version.parse(sf.__libsndfile_version__)
sf_required_version = packaging_version.parse("1.2.0")
sf_version_ok = sf_current_version >= sf_required_version


def numpy_to_ogg(audio_data: np.ndarray, sr: int, safe=True):
    """Convert the audio data to an in-memory ogg encoded file-like object using
    the soundfile library.

    Due to limitations in the available libraries for writing ogg files,
    numpy_to_ogg requires libsndfile 1.2.0 or greater to work reliably. When
    invoked, this function will check the version of libsndfile and raise an
    error if an older version is found. You can disable this check by calling
    with `safe=False`. For details, see:
    https://github.com/bastibe/python-soundfile/issues/130
    """

    # Why use the soundfile package instead of pydub?
    # feature gaps in pydub, ffmpeg, pip's soundfile, and libsndfile.

    # We could use pydub, but need to ensure that the underlying ffmpeg command
    # is compiled with vorbis support. This is not the case when installing with
    # `conda install ffmpeg` on an Intel Mac in May 2023.

    assert audio_data.ndim == 2, "audio_data must be 2 dimensional"
    channels, samples = audio_data.shape
    logging.info(
        "Creating ogg: length: {:.3f} seconds. Channels: {}".format(
            samples / sr, channels
        )
    )

    if not sf_version_ok:
        error_message = (
            f"Old libsndfile. Found:{sf_current_version} need:{sf_required_version}"
        )
        if safe:
            assert False, error_message
        else:
            logging.warning(error_message)

    in_memory_file_buffer = io.BytesIO()
    sf.write(in_memory_file_buffer, audio_data.T, sr, subtype="VORBIS", format="OGG")
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


def numpy_to_wav(audio_data: np.ndarray, sr: int, bit_depth=16):
    subtype = None
    if bit_depth == 8:
        subtype = "PCM_8"
    elif bit_depth == 16:
        subtype = "PCM_16"
    elif bit_depth == 24:
        subtype = "PCM_24"
    elif bit_depth == 32:
        subtype = "FLOAT"
    elif bit_depth == 64:
        subtype = "DOUBLE"
    else:
        raise ValueError(f"bit_depth must be 8, 16, 24, 32, or 64 (found {bit_depth}))")

    channels, samples = audio_data.shape
    logging.info(
        "Creating wav: length: {:.3f} seconds. Channels: {}".format(
            samples / sr, channels
        )
    )

    in_memory_file_buffer = io.BytesIO()
    sf.write(in_memory_file_buffer, audio_data.T, sr, format="WAV", subtype=subtype)
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


def numpy_to_file(numpy_data: np.ndarray):
    in_memory_file_buffer = io.BytesIO()
    np.save(in_memory_file_buffer, numpy_data)
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


def extract_wds_id_and_ext(readable_file):
    """Given an apache_beam ReadableFile, get a WebDataset style id and
    extension, where the id is everything up the first dot in the filename, and
    the extension is everything after the first dot in the filename.

    Can be used with beam.Map(extract_wds_id_and_ext)
    """
    path = pathlib.Path(readable_file.metadata.path)
    id = str(path.parent / path.name.split(".")[0])
    ext = ".".join(path.name.split(".")[1:])
    return (id, ext)


def write_file(output_path_and_buffer):
    """Helper function for writing a buffer to a given path. This should be able
    to handle gs:// style paths as well as local paths.

    Can be used with beam.Map(write_file)
    """
    output_path, buffer = output_path_and_buffer
    logging.info("Writing to: {}".format(output_path))
    with filesystems.FileSystems.create(output_path) as file_handle:
        file_handle.write(buffer.read())


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
        pass

    def process(self, readable_file):
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

        file_like = readable_file.open(mime_type="application/octet-stream")
        audio_tensor, sr = None, None

        # try loading the audio file with torchaudio, but catch RuntimeError,
        # which are thrown when torchaudio can't load the file.
        logging.info("Loading: {}".format(path))
        try:
            audio_tensor, sr = torchaudio.load(file_like, format=ext_without_dot)
        except RuntimeError:
            # We don't want to log the stacktrace, but for debugging, here's how
            # we could access it we can access it:
            #
            # import traceback
            # tb_str = traceback.format_exception(
            #     etype=type(e), value=e, tb=e.__traceback__
            # )
            logging.warning(f"Error loading audio: {path}")
            return []

        C, T = audio_tensor.shape
        logging.info(
            "Loaded {:.3f} second {}-channel audio: {}".format(T / sr, C, path)
        )

        return [(readable_file.metadata.path, audio_tensor, sr)]


class ResampleAudio(beam.DoFn):
    """Resample an audio to a new sample rate. Accepts a `(key, a, sr)` tuple
    were:

    - `key` is a string
    - `a` is a 2D torch.Tensor or numpy.ndarray with audio in the last dimension
    - `sr` is an int

    The return value will also be a `(key, a, sr)` tuple, but 'a' will always be
    torch.Tensor instance.
    """

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
        if self._source_sr_hint is not None:
            self.resample = torchaudio.transforms.Resample(
                self._source_sr_hint, self._target_sr
            )

    def process(self, audio_tuple: Tuple[str, Union[torch.Tensor, np.ndarray], int]):
        key, audio, source_sr = audio_tuple

        # check if audio_tensor is a numpy array or a torch tensor
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio)

        channels, _ = audio.shape
        if channels > 128:
            raise ValueError(
                f"audio_tensor ({key}) must have 128 or fewer channels (found {channels})"
            )

        resampled_audio: Optional[torch.Tensor] = None

        if source_sr == self._target_sr:
            resampled_audio = audio
            logging.info(
                f"Skipping resample because source was already ${self._target_sr}: {key}"
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


class SkipCompleted(beam.DoFn):
    def __init__(
        self,
        old_suffix: str,
        new_suffix: Union[str, List[str]],
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        check_timestamp: bool = False,
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

    def process(self, source_metadata: FileMetadata):
        check = remove_suffix(source_metadata.path, self._old_suffix)
        if self._source_dir is not None:
            check = move(check, self._source_dir, self._target_dir)
        checks = [check + suffix for suffix in self._new_suffixes]
        limits = [1 for _ in checks]

        results = FileSystems.match(checks, limits=limits)
        assert len(results) > 0, "Unexpected empty results. This should never happen."

        for result in results:
            num_matches = len(result.metadata_list)
            logging.info(f"Found {num_matches} of: {result.pattern}")
            if num_matches != 0 and self._check_timestamp:
                for target_metadata in result.metadata_list:
                    if (
                        target_metadata.last_updated_in_seconds
                        < source_metadata.last_updated_in_seconds
                    ):
                        logging.info(
                            f"Do not skip! A target was found ({target_metadata.path}), but it is older than source file ({source_metadata.path})"
                        )
                        return [source_metadata]
            elif num_matches == 0:
                return [source_metadata]

        logging.info(f"Targets already exist. Skipping: {source_metadata.path}")
        return []


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
        norm: float = torch.inf,
        device: Union[torch.device, str] = "cpu",
    ):
        self._audio_sr = audio_sr
        self._n_chroma = n_chroma
        self._n_fft = n_fft
        self._win_length = win_length
        self._hop_length = hop_length
        self._norm = norm
        self._device = device

    def setup(self):
        self._chroma_model = ChromaExtractor(
            sample_rate=self._audio_sr,
            n_chroma=self._n_chroma,
            n_fft=self._n_fft,
            win_length=self._win_length,
            hop_length=self._hop_length,
            norm=self._norm,
            device=self._device,
        )

    def process(self, element: Tuple[str, torch.Tensor, int]):
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
