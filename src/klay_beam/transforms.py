import pathlib
import io

import apache_beam as beam
from apache_beam.io import filesystems
import torchaudio
import pydub
import numpy as np


def numpy_to_mp3(audio_data: np.ndarray, sr: int, noisy=False):
    """Convert the audio tensor to an in-memory mp3 encoded file-like object.

    This needs to write a file-like object in memory. Loading file-like objects
    is now supported by torchaudio (both sox_io and soundfile backends), but
    torchaudio.save can only write to disk. As a result, we use pydub, which
    invokes ffmpeg to write to memory.

    For pytorch support see: https://pytorch.org/audio/stable/backend.html
    For pydub docs see: https://github.com/jiaaro/pydub
    """

    assert audio_data.ndim == 2, "audio_data must be 2 dimensional"
    samples, channels = audio_data.shape
    assert (
        samples > channels
    ), "incoming audio data must be interleaved (last dimension must be channels)"

    if noisy:
        print(
            "Creating mp3: length: {} seconds. Channels: {}".format(
                samples / sr, channels
            )
        )

    audio_data *= 2**15 - 1
    audio_data = audio_data.mean(axis=1)  # convert to mono
    audio_data = audio_data.astype(np.int16)

    raw_audio_buffer = io.BytesIO(audio_data.tobytes())

    audio_segment = pydub.AudioSegment.from_raw(
        raw_audio_buffer, frame_rate=sr, channels=1, sample_width=2
    )

    mp3_buffer = io.BytesIO()
    audio_segment.export(mp3_buffer, format="mp3")
    mp3_buffer.seek(0)
    return mp3_buffer


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
    print("Writing to {}".format(output_path))
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

        # get the file extension without a period in an safe way
        ext_without_dot = path.suffix.lstrip(".")
        ext_without_dot = None if ext_without_dot == "" else ext_without_dot

        file_like = readable_file.open(mime_type="application/octet-stream")
        audio_tensor, sr = None, None

        # try loading the audio file with torchaudio, but catch RuntimeError,
        # which are thrown when torchaudio can't load the file.
        try:
            audio_tensor, sr = torchaudio.load(file_like, format=ext_without_dot)
        except RuntimeError:
            # TODO handle/log this error
            print("Error loading file: {}".format(path))
            return []

        # Get a webdataset style id and key, where the id is everything up the
        # first dot in the filename, and the key is everything after the first
        # dot in the filename.
        id, ext = extract_wds_id_and_ext(readable_file)

        return [(id, ext, audio_tensor, sr)]
