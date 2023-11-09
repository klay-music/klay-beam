from audiotools import AudioSignal
from dac.utils import load_model
from dac.model import DAC
from dac.utils.encode import process as encode
from encodec import EncodecModel
from encodec.compress import compress_to_file as create_ecdc
from encodec.compress import decompress as decompress_ecdc
import logging
import torch
import io
from typing import Tuple, Optional, List, Any, Iterable
import copy

import apache_beam as beam

from klay_beam.torch_transforms import convert_audio
from klay_beam.path import remove_suffix
from klay_beam.utils import get_device


SAMPLE_RATE_MAP = {
    16000: "16khz",
    24000: "24khz",
    44100: "44khz",
    48000: "48khz",
}

VALID_EDITS = [
    'extract bass',
    'extract vocals',
    'extract drums',
    'extract other',
    'remove bass',
    'remove vocals',
    'remove drums',
    'remove other',
    'add bass',
    'add vocals',
    'add drums',
    'add other',
    'replace bass with vocals',
    'replace bass with drums',
    'replace bass with other',
    'replace vocals with bass',
    'replace vocals with drums',
    'replace vocals with other',
    'replace drums with bass',
    'replace drums with vocals',
    'replace drums with other',
    'replace other with bass',
    'replace other with vocals',
    'replace other with drums',
]


class ReadEncodec(beam.DoFn):
    def __init__(self, device: Optional[torch.device] = None):
        self._device = device

    def setup(self):
        if self._device is None:
            self._device = get_device()

    def process(self, element: Tuple[str, bytes]):
        key, file_like = element
        logging.info(f"Decoding ENCODEC: {key}")

        try:
            audio, sr = decompress_ecdc(file_like, self._device)
        except Exception as e:
            logging.error(f"Failed to decode ENCODEC: {key}")
            logging.error(e)
            return [beam.pvalue.TaggedOutput('failed', (key, e))]

        return [(key, audio, sr)]


class ExtractAtomicTriplets(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(self, t: int, tol: float=0.001, t_aug: bool=False, device: Optional[torch.device] = None):
        """
        t: length of audio in seconds
        tol: tolerance for silence to remove editing operations
        t_aug: whether to use time-located edit augmentations TODO: this is not implemented yet
        """
        self.t = t
        self._device = device
        self.tol = tol
        self.t_aug = t_aug
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
            self.edit_instructions = [x + f' from 0 seconds to {self.t // 3} seconds' for x in VALID_EDITS]
            self.edit_instructions += [x + f' from {self.t // 3} seconds to {2 * self.t // 3} seconds' for x in VALID_EDITS]
            self.edit_instructions += [x + f' from {2 * self.t // 3} seconds to {self.t} seconds' for x in VALID_EDITS]
        else:
            self.edit_instructions = VALID_EDITS

    def make_edit(self, edit: str, stem_d: dict, sr: int=48000):
        """
        edit: string, one of VALID_EDITS
        stem_d: dict, {stem_name: tensor}
        sr: int, sample rate
        """
        if self.t is None:
            times = [0, stem_d['source'].shape[-1] // sr]
        else:
            times = [0, self.t]
        if "extract" in edit:
            stem = edit.split(" ")[1]
            ref = stem_d["source"]
            tgt = torch.cat([ref[..., :times[0]*sr], stem_d[stem][..., times[0]*sr:times[1]*sr], ref[..., times[1]*sr:]], dim=-1)
        elif "remove" in edit:
            stem = edit.split(" ")[1]
            ref = stem_d["source"]
            tgt_all = sum([v for k,v in stem_d.items() if k not in [stem, 'source']])
            tgt = torch.cat([ref[..., :times[0]*sr], tgt_all[..., times[0]*sr:times[1]*sr], ref[..., times[1]*sr:]], dim=-1)
        elif "add" in edit:
            stem = edit.split(" ")[1]
            ref = sum([v for k,v in stem_d.items() if k not in [stem, 'source']])
            tgt = torch.cat([ref[..., :times[0]*sr], stem_d['source'][..., times[0]*sr:times[1]*sr], ref[..., times[1]*sr:]], dim=-1)
        elif "replace" in edit:
            stem1, stem2 = edit.split(" ")[1], edit.split(" ")[3]
            ref = sum([v for k,v in stem_d.items() if k not in [stem2, 'source']])
            tgt_all = sum([v for k,v in stem_d.items() if k not in [stem1, 'source']])
            tgt = torch.cat([ref[..., :times[0]*sr], tgt_all[..., times[0]*sr:times[1]*sr], ref[..., times[1]*sr:]], dim=-1)
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
        if self.t is None: # if t is None, then we pad to the longest stem
            max_len = max([x[1].shape[-1] for x in tracks])
            stem_d = {k.split(".")[1]: torch.cat([x[:, :max_len], torch.zeros(x.shape[0], max_len - x.shape[-1])], dim=-1) for k, x, sr in tracks}
        else:
            stem_d = {k.split(".")[1]: x[:, :self.t*sr] for k, x, sr in tracks}
        for k, v in stem_d.items():
            if v[..., :v.shape[1]//2000*2000].view(v.shape[0],v.shape[1] // 2000, 2000).pow(2).mean(-1).sqrt().mean() < self.tol:
                # print(f"WARNING: {k} is silent in {song}")
                edit_instructions = [x for x in edit_instructions if k not in x]
                stem_d[k] = None

        stem_d = {k:v for k,v in stem_d.items() if v is not None}
        edit_tupls = []
        for edit in edit_instructions:
            dp = self.make_edit(edit, stem_d, sr=sr)
            edit_tupls.append(dp)
        return [(song_n, path, edit_tupls)]