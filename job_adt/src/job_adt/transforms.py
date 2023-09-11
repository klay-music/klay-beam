import apache_beam as beam
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems
from io import BytesIO
import librosa
import logging

from magenta.common import tf_utils
from magenta.models.onsets_frames_transcription import (
    audio_label_data_utils,
    configs,
    constants,
    data,
    infer_util,
    train_util,
)
import note_seq
from note_seq import audio_io, midi_io, sequences_lib
import numpy as np
from pathlib import Path
import scipy
import tensorflow.compat.v1 as tf
from typing import Optional, Tuple, Union, List

from job_adt.utils import add_suffix, remove_suffix, write_file
from job_adt.path import move


tf.disable_v2_behavior()


class TranscribeDrumsAudio(beam.DoFn):
    def __init__(self, source_dir: str, checkpoint_dir: str):
        """
        Args:
            source_dir: str
                The directory where the audio files are stored. Trim this from the
                source filename
            checkpoint_dir : str
                The directory where the model checkpoint is stored.
        """
        self.source_dir = source_dir
        self.checkpoint_dir = checkpoint_dir

        self.config = configs.CONFIG_MAP["drums"]
        self.hparams = self.config.hparams
        self.hparams.batch_size = 1
        self.hparams.truncated_length_secs = 0


    def setup(self):
        # This will be executed only once when the pipeline starts.
        logging.info(
            f"Loading Onsets & Frames model for ADT from: {self.checkpoint_dir}"
        )

        # prepare dataset
        self.examples = tf.placeholder(tf.string, [None])
        self.dataset = data.provide_batch(
            examples=self.examples,
            preprocess_examples=True,
            params=self.hparams,
            is_training=False,
            shuffle_examples=False,
            skip_n_initial_records=0,
        )

    def process(
        self, loaded_audio_tuple: List[Tuple[str, bytes, int]]
    ) -> Tuple[str, Optional[note_seq.NoteSequence], int]:
        fname, wav_data, sr = loaded_audio_tuple[0]
        assert sr == 44_100, f"Expected 44.1k audio. Found {sr}. ({fname})"
        assert fname.endswith(".drums.wav"), f"Expected .drums.wav file. Found {fname}"
        logging.info(f"Found audio file: {fname} with length: {len(wav_data)} bytes.")

        out_filename = remove_suffix(fname, ".wav")

        try:
            # process_record returns an iterable so we need to unpack it
            to_process = []
            record = list(
                audio_label_data_utils.process_record(
                    wav_data=wav_data,
                    sample_rate=sr,
                    ns=note_seq.NoteSequence(),
                    example_id=fname,
                    min_length=0,
                    max_length=-1,
                    allow_empty_notesequence=True,
                )
            )
            to_process.append(record[0].SerializeToString())

            with tf.Session() as sess:
                # initialize
                sess.run(
                    [
                        tf.initializers.global_variables(),
                        tf.initializers.local_variables(),
                    ]
                )

                # prepare data iterator
                iterator = tf.data.make_initializable_iterator(self.dataset)
                examples = self.examples
                sess.run(iterator.initializer, feed_dict={examples: to_process})

                # prepare estimator
                estimator = train_util.create_estimator(
                    self.config.model_fn, self.checkpoint_dir, self.hparams
                )

                # predict
                next_record = iterator.get_next()

                def transcription_data(params):
                    del params
                    return tf.data.Dataset.from_tensors(sess.run(next_record))

                # run inference
                input_fn = infer_util.labels_to_features_wrapper(transcription_data)
                predictions = list(
                    estimator.predict(input_fn, yield_single_examples=False)
                )
                assert len(predictions) == 1, "Expected only one prediction"

                pred_note_sequence = note_seq.NoteSequence.FromString(
                    predictions[0]["sequence_predictions"][0]
                )
                return [(out_filename, pred_note_sequence)]

        except Exception as e:
            logging.error(f"Exception while performing ADT on: {fname}. Exception: {e}")
            return [(out_filename, None)]


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

    def __init__(self, target_sr: int):
        # This will be executed only once when the pipeline starts. This is
        # where you would create a lock or queue for global resources.
        self.target_sr = target_sr

    def process(self, readable_file):
        """
        Given an Apache Beam ReadableFile, return a `(input_filename, a, sr)` tuple where
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

        path = Path(readable_file.metadata.path)

        # get the file extension without a period in a safe way
        ext_without_dot = path.suffix.lstrip(".")
        ext_without_dot = None if ext_without_dot == "" else ext_without_dot

        file_like = readable_file.open(mime_type="application/octet-stream")
        audio_array = None

        logging.info("Loading: {}".format(path))
        try:
            audio_array, sr = librosa.load(file_like, sr=self.target_sr, mono=True)
            assert sr == self.target_sr
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
