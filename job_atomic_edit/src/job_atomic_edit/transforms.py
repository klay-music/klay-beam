import logging
import torch
from typing import Tuple, Optional, List, Any, Iterable, Union
import copy
from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

import apache_beam as beam

from klay_beam.torch_transforms import convert_audio
from klay_beam.path import remove_suffix, move
from klay_beam.utils import get_device


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
                        found_any += 1
            elif num_matches == 0:
                found_any += 1
        if found_any == 0:  # i.e. if all targets already exist
            logging.info(f"Targets already exist. Skipping: {source_metadata.path}")
            return []
        elif found_any < len(checks):  # i.e. if some targets already exist but not all
            logging.info(
                f"Some targets already exist. Assuming missing due to silent tracks. Skipping: {source_metadata.path}"
            )
            return []
        else:  # i.e. if no targets already exist
            return [source_metadata]


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
        self.POSSIBLE_STEMS = ["bass", "drums", "other", "vocals", 'source']
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
            if tgt_all == 0:
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
            if ref == 0:
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
            if ref == 0 or tgt_all == 0:
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
        sr = tracks[0][-1]

        assert type(tracks[0]) == tuple, "tracks should be a list of tuples"
        assert type(tracks[0][0]) == str
        assert type(tracks[0][1]) == torch.Tensor
        assert type(tracks[0][2]) == int
        assert all([x[-1] == sr for x in tracks])

        # note that path here is the folder name, while song_n just extracts the song name
        # for example:
        # full path = gs://klay-datasets-001/mtg-jamendo-90s-crop/00/1002000.bass.wav
        # song_n = 1002000
        # path = gs://klay-datasets-001/mtg-jamendo-90s-crop/00/1002000
        path = tracks[0][0].split(".")[0]
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

        if stem_d.get('source') is None:
            stem_d['source'] = sum([v for k, v in stem_d.items() if k not in ['source']])

        

        for edit in edit_instructions:
            dp = self.make_edit(edit, stem_d, sr=sr)
            edit_tupls.append(dp)
        edit_tupls = [x for x in edit_tupls if x is not None]
        return [(song_n, path, edit_tupls, sr)]
