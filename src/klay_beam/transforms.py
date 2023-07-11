import pathlib
import io
import traceback
from typing import Optional, Type
from packaging import version as packaging_version

import apache_beam as beam
from apache_beam.io import filesystems
import torchaudio
import pydub
import soundfile as sf
import numpy as np
import logging


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
    audio_data *= 2 ** (bit_depth - 1) - 1  # scale to bit_depth.

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

    # `pip install soundfile` has buggy vorbis support. soundfile can only
    # reliably write ogg files when the underlying libsndfile is 1.2.0 or greater
    # You can check this with `soundfile.__libsndfile_version__`. For details,
    # see: https://github.com/bastibe/python-soundfile/issues/130

    assert audio_data.ndim == 2, "audio_data must be 2 dimensional"
    channels, samples = audio_data.shape
    logging.info(
        "Creating ogg: length: {:.3f} seconds. Channels: {}".format(
            samples / sr, channels
        )
    )

    current_version = packaging_version.parse(sf.__libsndfile_version__)
    required_version = packaging_version.parse("1.2.0")

    if safe:
        assert (
            current_version >= required_version
        ), f"Out of date libsndfile. Found:{current_version} needed:{required_version} or greater."

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
    else:
        raise ValueError(f"bit_depth must be 8, 16, or 24 (found {bit_depth}))")

    in_memory_file_buffer = io.BytesIO()
    sf.write(in_memory_file_buffer, audio_data.T, sr, format="WAV", subtype=subtype)
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
    logging.info("Writing to {}".format(output_path))
    with filesystems.FileSystems.create(output_path) as file_handle:
        file_handle.write(buffer.read())


class LoadWithTorchaudio(beam.DoFn):
    """Use torchaudio to load audio files to tensors

    Note that torchaudio depends on libavcodec, which can be installed with:
    `conda install 'ffmpeg<5'`.

    See: https://github.com/pytorch/audio/issues/2363#issuecomment-1179089175

    Note that generally, custom  functions have a few requirements that help
    them work well in on distributed runners. They are:
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
        Given an Apache Beam ReadableFile, return a `(id, key, a, sr)` tuple where
            - `id` is a string
            - `key` is string
            - `a` is a pytorch Tensor
            - `sr` is an int

        For a stereo audio file named '/path/to.some/file.key.mp3', return
        ```
        (
            '/path/to.some/file',
            'key.mp3',
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
        except RuntimeError as e:
            # TODO handle/log this error
            #
            # July 2023: It's not the best practice to put a stacktrace in a log
            # message. For now, I need just a little bit more information when
            # this fails
            tb_str = traceback.format_exception(
                etype=type(e), value=e, tb=e.__traceback__
            )
            logging.warning("Error loading file: {}\n{}".format(path, tb_str))
            return []
        C, T = audio_tensor.shape
        logging.info("Loaded {:.3f} second {}-channel file: {}".format(T / sr, C, path))
        # Get a WebDataset style id and key, where the id is everything up the
        # first dot in the filename, and the key is everything after the first
        # dot in the filename.
        id, ext = extract_wds_id_and_ext(readable_file)

        return [(id, ext, audio_tensor, sr)]
