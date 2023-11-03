import argparse
import os.path
import logging
from typing import Optional, Type, Union, List
import io
import apache_beam as beam
import apache_beam.io.fileio as beam_io
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems
from apache_beam.options.pipeline_options import (
    PipelineOptions,
    SetupOptions,
    StandardOptions,
    WorkerOptions,
)
import torch
import torchaudio

from klay_beam.transforms import *

from klay_beam.torch_transforms import *
from job_atomic_edit.transforms import ExtractDAC, ExtractEncodec, ExtractEncodecGrouped, ExtractAtomicTriplets, VALID_EDITS

class SkipCompletedGrouped(beam.DoFn):
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

    def process_track(self, source_metadata: FileMetadata):
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
    
    def process(self, src: tuple):
        song, source_metadatas = src
        return [(song, [item for source_metadata in source_metadatas for item in self.process_track(source_metadata)])]

class LoadWithTorchaudioGrouped(beam.DoFn):
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

    def process_track(self, readable_file: beam_io.ReadableFile):
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

    def process(self, src: tuple):
        song, source_metadatas = src
        return [(song, [item for source_metadata in source_metadatas for item in self.process_track(source_metadata)])]
"""
Job for extracting EnCodec features. See job_atomic_edit/README.md for details.
"""
def torch_to_file(torch_data: torch.Tensor, sample_rate: int):
    in_memory_file_buffer = io.BytesIO()
    torchaudio.save(in_memory_file_buffer, torch_data,sample_rate=sample_rate, format="wav")
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer

edit2ix = {x:i for i,x in enumerate(VALID_EDITS)}
ix2st = {
    0: 'src',
    2: 'tgt'
}


DEFAULT_IMAGE = "us-docker.pkg.dev/klay-home/klay-docker/klay-beam:0.11.0-py3.10-beam2.51.0-torch2.0"

class UngroupElements(beam.DoFn):
    
    def __init__(self, sample_rate: int):
        assert sample_rate in [
            24000,
            48000,
        ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.sample_rate = sample_rate


    def process(self, element):
        k, path, v = element
        for elem in list(v):
            # process your element 
            for ix, el in enumerate(elem):
                if type(el) == torch.Tensor:
                    yield (f"{path}.{ix2st[ix]}.{edit2ix[elem[1]]}", el, self.sample_rate)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--source_audio_path",
        dest="input",
        required=True,
        help="""
        Specify the parent audio file directory. This can be a local path or a gs:// URI.

        To get only some wav files, try:
        '/Users/alice/datasets/fma_large/005/'

        To run on the full dataset use:
        'gs://klay-datasets/mtg_jamendo_autotagging/audios/'
        """,
    )

    parser.add_argument(
        "--nac_name",
        required=True,
        choices=["dac", "encodec"],
        help="""
        Which neural audio codec should we use? Options are ['dac' or 'encodec']
        """,
    )

    parser.add_argument(
        "--nac_input_sr",
        required=True,
        type=int,
        choices=[16000, 24000, 44100, 48000],
        help="""
        Which audio sample rate should we extract from?
        """,
    )
    parser.add_argument(
        "--t",
        required=False,
        type=int,
        default=None,
        help="""
        Which audio sample rate should we extract from?
        """,
    )

    parser.add_argument(
        "--audio_suffix",
        required=True,
        choices=[".mp3", ".wav", ".aif", ".aiff"],
        help="""
        Which audio file extension to search for when scanning input dir?
        """,
    )

    return parser.parse_known_args(None)


def run():
    known_args, pipeline_args = parse_args()
    print("known_args: {}".format(known_args))
    print("pipeline_args: {}".format(pipeline_args))

    # pickle the main session in case there are global objects
    pipeline_options = PipelineOptions(pipeline_args)
    pipeline_options.view_as(SetupOptions).save_main_session = True

    # Set the default docker image if we're running on Dataflow
    if (
        pipeline_options.view_as(StandardOptions).runner == "DataflowRunner"
        and pipeline_options.view_as(WorkerOptions).sdk_container_image is None
    ):
        pipeline_options.view_as(WorkerOptions).sdk_container_image = DEFAULT_IMAGE

    # Pattern to recursively find audio files inside source_audio_path
    match_pattern = os.path.join(known_args.input, f"**{known_args.audio_suffix}")

    # instantiate NAC extractor here so we can use computed variables
    edit_fn = ExtractAtomicTriplets(known_args.t)
    ungroup_fn = UngroupElements(known_args.nac_input_sr)
    extract_fn: Union[ExtractDAC, ExtractEncodec, ExtractEncodecGrouped]
    if known_args.nac_name == "dac":
        extract_fn = ExtractDAC(known_args.nac_input_sr)
    elif known_args.nac_name == "encodec":
        extract_fn = ExtractEncodec(known_args.nac_input_sr)

    with beam.Pipeline(argv=pipeline_args, options=pipeline_options) as p:
        audio_files = (
            p
            # MatchFiles produces a PCollection of FileMetadata objects
            | beam_io.MatchFiles(match_pattern)
            # Prevent "fusion". See:
            # https://cloud.google.com/dataflow/docs/pipeline-lifecycle#preventing_fusion
            | beam.Reshuffle()
            
            | "SkipCompleted"
            >> beam.ParDo(
                SkipCompleted(
                    old_suffix=known_args.audio_suffix,
                    new_suffix=extract_fn.suffix,
                    check_timestamp=True,
                )
            )
            # # ReadMatches produces a PCollection of ReadableFile objects
            | beam_io.ReadMatches()
            # | beam.Map(lambda x: (x.metadata.path.split("/")[-1].split(".")[0], x))
            # | "Group by track" >> beam.GroupByKey()
            # | beam.Map(print)
            | "LoadAudio" >> beam.ParDo(LoadWithTorchaudio())
            # | beam.Map(print)
        )

        out = (
            audio_files
            | f"Resample: {extract_fn.sample_rate}Hz"
            >> beam.ParDo(
                ResampleTorchaudioTensor(
                    source_sr_hint=48_000,
                    target_sr=extract_fn.sample_rate,
                )
            )
            | beam.Map(lambda x: (x[0].split("/")[-1].split(".")[0], x))
            | "Group by track" >> beam.GroupByKey()
            | "Get Edit Triplets" >> beam.ParDo(edit_fn)
            | "Ungroup Elements" >> beam.ParDo(ungroup_fn)
            # | "ExtractNAC" >> beam.ParDo(extract_fn).with_outputs("ecdc", main="npy")
            # | beam.Map(print)
        )
        # write out wav files
        (
            out | "CreatewavFile" >> beam.Map(lambda x: (x[0] + '.wav', torch_to_file(x[1], x[2])))
            | "PersistFile" >> beam.Map(write_file)
        )



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run()
