import pathlib
import apache_beam as beam
import apache_beam.io.fileio
import torchaudio
import pydub
import io

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
        id = str(path.parent / path.name.split(".")[0])
        key = ".".join(path.name.split(".")[1:])

        return [(id, key, audio_tensor, sr)]

class WriteTorchaudio(beam.DoFn):
    """Uses convert audio tensors to data.

    This needs to write a file-like object in memory. Loading file-like objects
    is supported by torchaudio (both sox_io and soundfile backends), but
    torchaudio.save can only write to disk. As a result, we use pydub, which
    uses ffmpeg.

    For pytorch support see: https://pytorch.org/audio/stable/backend.html
    For pydub docs see: https://github.com/jiaaro/pydub
    """

    def process(self, element):
        id, key, audio_tensor, sr = element

        # convert the audio tensor to an in-memory mp3 encoded file-like object using pydub

        audio_data = audio_tensor.numpy()
        audio_data *= 32767
        audio_data = audio_data.astype(np.int16)
        audio_data_mono = audio_data.mean(axis=1)
        raw_audio_buffer = io.BytesIO(audio_data_mono.tobytes())

        audio_segment = pydub.AudioSegment.from_raw(raw_audio_buffer, frame_rate=sr, channels=1, sample_width=2)

        mp3_buffer = io.BytesIO()
        audio_segment.export(mp3_buffer, format="mp3")
        mp3_buffer.seek(0)

        return [(id, key, mp3_buffer)]
