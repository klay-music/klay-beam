import pathlib
import io
from typing import Optional, Type, Union, List
from packaging import version as packaging_version
import apache_beam as beam
from apache_beam.io import filesystems
import apache_beam.io.fileio as beam_io
import pydub
import soundfile as sf
import numpy as np
import logging
import librosa

from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

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
                            f"Do not skip! A target was found ({target_metadata.path}), but it is "
                            f"older than source file ({source_metadata.path})"
                        )
                        return [source_metadata]
            elif num_matches == 0:
                return [source_metadata]

        logging.info(f"Targets already exist. Skipping: {source_metadata.path}")
        return []


class MultiMatchFiles(beam.PTransform):
    """Like beam.io.fileio.MatchFiles, but takes a list of patterns and returns
    a single PCollection of FileMetadata objects."""

    def __init__(self, patterns: list[str]):
        self.patterns = patterns

    def expand(self, pcoll: beam.PCollection):
        # Create a list of PCollections by matching each pattern
        matched_files_list = [
            pcoll.pipeline | f"MatchFiles {pattern}" >> beam_io.MatchFiles(pattern)
            for pattern in self.patterns
        ]

        # Flatten the list of PCollections into a single PCollection
        return matched_files_list | "Flatten File Results" >> beam.Flatten()


class LoadWithLibrosa(beam.DoFn):
    """Use librosa to load audio files to numpy arrays.

    NOTES:

    Note that generally, custom functions have a few requirements that help them
    work well in on distributed runners. They are:
        - The function should be thread-compatible
        - The function should be serializable
        - Recommended: the function be idempotent

    For details about these requirements, see the Apache Beam documentation:
    https://beam.apache.org/documentation/programming-guide/#requirements-for-writing-user-code-for-beam-transforms
    """

    def __init__(self, target_sr: Optional[int], mono: bool):
        self.target_sr = target_sr
        self.mono = mono

    def process(self, readable_file: beam_io.ReadableFile):
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, a, sr)` 
        tuple where:
            - `input_filename` is a string
            - `a` is a numpy array
            - `sr` is an int

        For a stereo audio file named '/path/to.some/file.key.mp3', return
        ```
        (
            '/path/to.some/file.key.mp3',
            np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.3]]),
            44100
        )
        ```

        WARNING! Librosa is inconsistent about the shape of audio array. When
        mono=True, AND when the audio file is a mono file, librosa.load
        returns a 1-D numpy array. Librosa ONLY returns a 2-D array when the
        input audio file has multiple channels AND mono=False.
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
        audio_array = None

        logging.info("Loading: {}".format(path))
        try:
            audio_array, sr = librosa.load(file_like, sr=self.target_sr, mono=self.mono)
            if self.target_sr is not None: assert sr == self.target_sr
        except RuntimeError:
            # We don't want to log the stacktrace, but for debugging, here's how
            # we could access it we can access it:
            #
            # import traceback
            # tb_str = traceback.format_exception(
            #     etype=type(e), value=e, tb=e.__traceback__
            # )
            logging.warning(f"Error loading audio: {path}")
            return (readable_file.metadata.path, np.ndarray([]), sr)

        logging.info(f"Loaded {len(audio_array) / sr:.3f} second mono audio: {path}")

        return [(readable_file.metadata.path, audio_array, sr)]
