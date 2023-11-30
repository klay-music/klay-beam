import logging
import torch
from typing import Tuple, Optional, List, Any, Iterable, Union
import copy
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

import apache_beam as beam

from klay_beam.path import remove_suffix, move
from klay_beam.utils import get_device
import pathlib
import torchaudio
from klay_beam.torch_utils import ensure_torch_available


SAMPLE_RATE_MAP = {
    16000: "16khz",
    24000: "24khz",
    44100: "44khz",
    48000: "48khz",
}

VALID_EDITS = [
    "extract bass",
    "extract vocals",
    "extract drums",
    "extract other",
    "remove bass",
    "remove vocals",
    "remove drums",
    "remove other",
    "add bass",
    "add vocals",
    "add drums",
    "add other",
    "replace bass with vocals",
    "replace bass with drums",
    "replace bass with other",
    "replace vocals with bass",
    "replace vocals with drums",
    "replace vocals with other",
    "replace drums with bass",
    "replace drums with vocals",
    "replace drums with other",
    "replace other with bass",
    "replace other with vocals",
    "replace other with drums",
]


class SkipCompletedMulti(beam.DoFn):
    """
    This is a beam DoFn that checks if any of the target triplets already exist.
    """

    def __init__(
        self,
        old_suffix: Union[str, List[str]],
        new_suffix: Union[str, List[str]],
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        check_timestamp: bool = False,
    ):
        if isinstance(new_suffix, str):
            new_suffix = [new_suffix]
        if isinstance(old_suffix, str):
            old_suffix = [old_suffix]
        self._new_suffixes = new_suffix
        self._old_suffixes = old_suffix

        assert (source_dir is None) == (
            target_dir is None
        ), "source_dir and target_dir must both be None or strings"

        self._source_dir = source_dir
        self._target_dir = target_dir
        self._check_timestamp = check_timestamp

    def process(self, source_metadata: FileMetadata):
        # check which suffix the file has
        for old_suffix in self._old_suffixes:
            if source_metadata.path.endswith(old_suffix):
                tgt_suffix = old_suffix
                break
        check = remove_suffix(source_metadata.path, tgt_suffix)
        if self._source_dir is not None:
            check = move(check, self._source_dir, self._target_dir)
        checks = [check + suffix for suffix in self._new_suffixes]
        limits = [1 for _ in checks]

        results = FileSystems.match(checks, limits=limits)
        assert len(results) > 0, "Unexpected empty results. This should never happen."
        found_any = 0  # counter for how many of the targets already exist
        for result in results:
            num_matches = len(result.metadata_list)
            # logging.info(f"Found {num_matches} of: {result.pattern}")
            if num_matches != 0 and self._check_timestamp:
                for target_metadata in result.metadata_list:
                    if (
                        target_metadata.last_updated_in_seconds
                        < source_metadata.last_updated_in_seconds
                    ):
                        # logging.info(
                        #     f"Do not skip! A target was found ({target_metadata.path}), but it is "
                        #     f"older than source file ({source_metadata.path})"
                        # )
                        found_any += 1
            elif num_matches == 0:
                found_any += 1
        if found_any == 0:  # i.e. if all targets already exist
            # logging.info(f"Targets already exist. Skipping: {source_metadata.path}")
            return []
        elif found_any < len(checks):  # i.e. if some targets already exist but not all
            # logging.info(
            #     f"Some targets already exist. Assuming missing due to silent tracks. Skipping: {source_metadata.path}"
            # )
            return []
        else:  # i.e. if no targets already exist
            return [source_metadata]


class _MutliReadMatchesFn(beam.DoFn):
    def __init__(self, compression, skip_directories):
        self._compression = compression
        self._skip_directories = skip_directories

    def process_single(
        self,
        file_metadata: Union[str, beam.io.filesystem.FileMetadata],
    ) -> Iterable[beam.io.fileio.ReadableFile]:
        metadata = (
            beam.io.filesystem.FileMetadata(file_metadata, 0)
            if isinstance(file_metadata, str)
            else file_metadata
        )

        if (
            metadata.path.endswith("/") or metadata.path.endswith("\\")
        ) and self._skip_directories:
            return
        elif metadata.path.endswith("/") or metadata.path.endswith("\\"):
            raise beam.io.filesystem.BeamIOError(
                "Directories are not allowed in ReadMatches transform."
                "Found %s." % metadata.path
            )

        # TODO: Mime type? Other arguments? Maybe arguments passed in to transform?
        return beam.io.fileio.ReadableFile(metadata, self._compression)

    def process(
        self,
        file_metadatas: Tuple[
            Any, Iterable[Union[str, beam.io.filesystem.FileMetadata]]
        ],
    ) -> Tuple[Any, Iterable[Iterable[beam.io.fileio.ReadableFile]]]:
        yield file_metadatas[0], [self.process_single(f) for f in file_metadatas[1]]


class MultiReadMatches(beam.PTransform):
    """Converts each result of MatchFiles() or MatchAll() to a ReadableFile.

    This helps read in a file's contents or obtain a file descriptor."""

    def __init__(self, compression=None, skip_directories=True):
        self._compression = compression
        self._skip_directories = skip_directories

    def expand(
        self,
        pcolls: beam.PCollection[
            Tuple[Any, Iterable[Union[str, beam.io.filesystem.FileMetadata]]]
        ],
    ):
        return pcolls | beam.ParDo(
            _MutliReadMatchesFn(self._compression, self._skip_directories)
        )


class _MultiLoadWithTorchaudio(beam.DoFn):
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

    def process_single(self, readable_file: beam.io.fileio.ReadableFile):
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

        return readable_file.metadata.path, audio_tensor, sr
        # beam.pvalue.TaggedOutput("duration_seconds", duration_seconds),
        # ]

    def process(
        self,
        readable_files: Tuple[Any, Iterable[beam.io.fileio.ReadableFile]],
    ) -> Tuple[Any, Iterable[Any]]:
        yield (readable_files[0], [self.process_single(f) for f in readable_files[1]])


class MultiLoadWithTorchaudio(beam.PTransform):
    def __init__(self):
        self.func = _MultiLoadWithTorchaudio()

    def expand(self, element: Tuple[Any, Iterable[Any]]) -> Tuple[Any, Iterable[Any]]:
        return element | beam.ParDo(self.func)


class ExtractAtomicTriplets(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(
        self,
        t: int,
        tol: float = 0.001,
        t_aug: bool = False,
        device: Optional[torch.device] = None,
    ):
        """
        t: length of audio in seconds
        tol: tolerance for silence to remove editing operations
        t_aug: whether to use time-located edit augmentations TODO: this is not implemented yet
        """
        self.t = t
        self._device = device
        self.tol = tol
        self.t_aug = t_aug
        self.POSSIBLE_STEMS = ["bass", "drums", "other", "vocals", "source"]
        if self.t_aug == True:
            raise NotImplementedError("t_aug is not implemented yet")

    def setup(self):
        """
        Namely, this will setup the beam class with the correct set of edits to apply.
        """
        if self._device is None:
            self._device = get_device()
        # add edit instructions for start to t // 3, t // 3 to 2t // 3, 2t // 3 to end
        if self.t_aug:
            self.edit_instructions = [
                x + f" from 0 seconds to {self.t // 3} seconds" for x in VALID_EDITS
            ]
            self.edit_instructions += [
                x + f" from {self.t // 3} seconds to {2 * self.t // 3} seconds"
                for x in VALID_EDITS
            ]
            self.edit_instructions += [
                x + f" from {2 * self.t // 3} seconds to {self.t} seconds"
                for x in VALID_EDITS
            ]
        else:
            self.edit_instructions = VALID_EDITS

    def make_edit(self, edit: str, stem_d: dict, sr: int = 48000):
        """
        edit: string, one of VALID_EDITS
        stem_d: dict, {stem_name: tensor}
        sr: int, sample rate
        """
        if self.t is None:
            times = [0, stem_d["source"].shape[-1] // sr]
        else:
            times = [0, self.t]
        if "extract" in edit:
            stem = edit.split(" ")[1]
            ref = stem_d["source"]
            tgt = torch.cat(
                [
                    ref[..., : times[0] * sr],
                    stem_d[stem][..., times[0] * sr : times[1] * sr],
                    ref[..., times[1] * sr :],
                ],
                dim=-1,
            )
        elif "remove" in edit:
            stem = edit.split(" ")[1]
            ref = stem_d["source"]
            tgt_all = sum([v for k, v in stem_d.items() if k not in [stem, "source"]])
            if type(tgt_all) == int:
                return None
            tgt = torch.cat(
                [
                    ref[..., : times[0] * sr],
                    tgt_all[..., times[0] * sr : times[1] * sr],
                    ref[..., times[1] * sr :],
                ],
                dim=-1,
            )
        elif "add" in edit:
            stem = edit.split(" ")[1]
            ref = sum([v for k, v in stem_d.items() if k not in [stem, "source"]])
            if type(ref) == int:
                return None
            tgt = torch.cat(
                [
                    ref[..., : times[0] * sr],
                    stem_d["source"][..., times[0] * sr : times[1] * sr],
                    ref[..., times[1] * sr :],
                ],
                dim=-1,
            )
        elif "replace" in edit:
            stem1, stem2 = edit.split(" ")[1], edit.split(" ")[3]
            ref = sum([v for k, v in stem_d.items() if k not in [stem2, "source"]])
            tgt_all = sum([v for k, v in stem_d.items() if k not in [stem1, "source"]])
            if type(ref) == int or type(tgt_all) == int:
                return None
            tgt = torch.cat(
                [
                    ref[..., : times[0] * sr],
                    tgt_all[..., times[0] * sr : times[1] * sr],
                    ref[..., times[1] * sr :],
                ],
                dim=-1,
            )
        return ref, edit, tgt

    def process(self, element: Tuple[Any, Iterable[Any]]):
        song_n, tracks = element
        edit_instructions = copy.copy(self.edit_instructions)
        # under the hood, tracks may be converted to a _ConcatSequence dtype, which is not indexable
        # thus, accessing sr and path uses the list comprehension logic below
        sr = [x[-1] for x in tracks][0]

        # note that path here is the folder name, while song_n just extracts the song name
        # for example:
        # full path = gs://klay-datasets-001/mtg-jamendo-90s-crop/00/1002000.bass.wav
        # song_n = 1002000
        # path = gs://klay-datasets-001/mtg-jamendo-90s-crop/00/1002000
        # path = tracks[0][0].split(".")[0]
        path = [x[0].split(".")[0] for x in tracks][0]
        if self.t is None:  # if t is None, then we pad to the longest stem
            max_len = max([x[1].shape[-1] for x in tracks])
            stem_d = {
                track_id.split(".")[1]: torch.cat(
                    [
                        track[:, :max_len],
                        torch.zeros(track.shape[0], max_len - track.shape[-1]),
                    ],
                    dim=-1,
                )
                for track_id, track, sr in tracks
            }
        else:
            stem_d = {k.split(".")[1]: x[:, : self.t * sr] for k, x, sr in tracks}
        for track_id, track in stem_d.items():
            # first check for silent tracks by calculating RMS amplitude
            if (
                track[..., : track.shape[1] // 2000 * 2000]
                .view(track.shape[0], track.shape[1] // 2000, 2000)
                .pow(2)
                .mean(-1)
                .sqrt()
                .mean()
                < self.tol
            ):
                # print(f"WARNING: {k} is silent in {song}")
                # if silent, remove all edits that involve this stem
                edit_instructions = [x for x in edit_instructions if track_id not in x]
                stem_d[track_id] = None

        stem_d = {
            track_id: track for track_id, track in stem_d.items() if track is not None
        }
        missing_stems = set(self.POSSIBLE_STEMS) - set(stem_d.keys())
        for stem in missing_stems:
            # remove edits that involve missing stems
            edit_instructions = [x for x in edit_instructions if stem not in x]
        edit_tupls = []

        if stem_d.get("source") is None:
            stem_d["source"] = sum(
                [v for k, v in stem_d.items() if k not in ["source"]]
            )

        for edit in edit_instructions:
            dp = self.make_edit(edit, stem_d, sr=sr)
            edit_tupls.append(dp)
        edit_tupls = [x for x in edit_tupls if x is not None]
        return [(song_n, path, edit_tupls, sr)]
