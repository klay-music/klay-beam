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
from typing import Tuple, Optional
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


class ExtractEncodec(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        assert sample_rate in [
            24000,
            48000,
        ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        if self._device is None:
            self._device = get_device()
        if self.sample_rate == 24000:
            self.codec = EncodecModel.encodec_model_24khz()
        elif self.sample_rate == 48000:
            self.codec = EncodecModel.encodec_model_48khz()

        self.codec.eval()
        self.codec.to(self._device)

    @property
    def output_file_format(self) -> str:
        if self.sample_rate == 24000:
            return "npy"
        elif self.sample_rate == 48000:
            return "ecdc"

    @property
    def suffix(self) -> str:
        return f".encodec_{SAMPLE_RATE_MAP[self.sample_rate]}.{self.output_file_format}"

    @property
    def num_channels(self) -> int:
        return 2 if self.sample_rate == 48000 else 1

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav")
        output_filename = remove_suffix(output_filename, ".mp3")
        output_filename += self.suffix

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.sample_rate,
            target_channels=self.num_channels,
        )
        audio_batch = audio.unsqueeze(0)

        with torch.no_grad():
            if self.output_file_format == "ecdc":
                file_like = io.BytesIO()
                create_ecdc(self.codec, audio, file_like, use_lm=False)
                file_like.seek(0)
                logging.info(f"Encoded ecdc with ENCODEC: {output_filename}")
                return [beam.pvalue.TaggedOutput('ecdc', (output_filename, file_like))]

            # The Encodec format is designed to be decoded live, so the channels
            # must be interleaved. Each "Frame" should be a tuple of (codebook, scale)
            frames = self.codec.encode(audio_batch)

            # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
            # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
            tensors = [t[0] for t in frames] # remove the "scale" from each frame
            codes = torch.cat(tensors, dim=2) # shape: [B, K, T]
            codes = codes.detach().cpu().numpy()

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with ENCODEC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]


class ExtractEncodecGrouped(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(self, sample_rate: int, device: Optional[torch.device] = None):
        assert sample_rate in [
            24000,
            48000,
        ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        if self._device is None:
            self._device = get_device()
        if self.sample_rate == 24000:
            self.codec = EncodecModel.encodec_model_24khz()
        elif self.sample_rate == 48000:
            self.codec = EncodecModel.encodec_model_48khz()

        self.codec.eval()
        self.codec.to(self._device)

    @property
    def output_file_format(self) -> str:
        if self.sample_rate == 24000:
            return "npy"
        elif self.sample_rate == 48000:
            return "ecdc"

    @property
    def suffix(self) -> str:
        return f".encodec_{SAMPLE_RATE_MAP[self.sample_rate]}.{self.output_file_format}"

    @property
    def num_channels(self) -> int:
        return 2 if self.sample_rate == 48000 else 1

    def process_track(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav")
        output_filename = remove_suffix(output_filename, ".mp3")
        output_filename += self.suffix

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.sample_rate,
            target_channels=self.num_channels,
        )
        audio_batch = audio.unsqueeze(0)

        with torch.no_grad():
            if self.output_file_format == "ecdc":
                file_like = io.BytesIO()
                create_ecdc(self.codec, audio, file_like, use_lm=False)
                file_like.seek(0)
                logging.info(f"Encoded ecdc with ENCODEC: {output_filename}")
                return [beam.pvalue.TaggedOutput('ecdc', (output_filename, file_like))]

            # The Encodec format is designed to be decoded live, so the channels
            # must be interleaved. Each "Frame" should be a tuple of (codebook, scale)
            frames = self.codec.encode(audio_batch)

            # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
            # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
            tensors = [t[0] for t in frames] # remove the "scale" from each frame
            codes = torch.cat(tensors, dim=2) # shape: [B, K, T]
            codes = codes.detach().cpu().numpy()

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with ENCODEC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]
    
    def process(self, src: tuple):
        song, elements = src
        return [(song, [item for element in elements for item in self.process_track(element)])]


class ExtractAtomicTriplets(beam.DoFn):
    """Beam DoFn for extracting encodec tokens from audio."""

    def __init__(self, t: int, tol: float=0.001, t_aug: bool=False, device: Optional[torch.device] = None):
        # assert sample_rate in [
        #     24000,
        #     48000,
        # ], f"Invalid sample_rate: {sample_rate} for encodec model"
        self.t = t
        self._device = device
        self.tol = tol
        self.t_aug = t_aug

    def setup(self):
        if self._device is None:
            self._device = get_device()
        # add edit instructions for start to t // 3, t // 3 to 2t // 3, 2t // 3 to end
        if self.t_aug:
            self.edit_instructions = [x + f' from 0 seconds to {self.t // 3} seconds' for x in VALID_EDITS]
            self.edit_instructions += [x + f' from {self.t // 3} seconds to {2 * self.t // 3} seconds' for x in VALID_EDITS]
            self.edit_instructions += [x + f' from {2 * self.t // 3} seconds to {self.t} seconds' for x in VALID_EDITS]
        else:
            self.edit_instructions = VALID_EDITS
        # if self.sample_rate == 24000:
        #     self.codec = EncodecModel.encodec_model_24khz()
        # elif self.sample_rate == 48000:
        #     self.codec = EncodecModel.encodec_model_48khz()

        # self.codec.eval()
        # self.codec.to(self._device)

    @property
    def output_file_format(self) -> str:
        if self.sample_rate == 24000:
            return "npy"
        elif self.sample_rate == 48000:
            return "ecdc"

    @property
    def suffix(self) -> str:
        return f".encodec_{SAMPLE_RATE_MAP[self.sample_rate]}.{self.output_file_format}"

    @property
    def num_channels(self) -> int:
        return 2 if self.sample_rate == 48000 else 1
    
    def make_edit(self, edit, stem_d, sr=48000):
        # lots of if statements
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

    def process(self, element):
        song_n, tracks = element
        edit_instructions = copy.copy(self.edit_instructions)
        # key, x, source_sr = element
        sr = tracks[0][-1]
        assert all([x[-1] == sr for x in tracks])
        path = tracks[0][0].split(".")[0]
        if self.t is None:
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
        # # Ensure that we are naming the file correctly.
        # output_filename = remove_suffix(key, ".wav")
        # output_filename = remove_suffix(output_filename, ".mp3")
        # output_filename += self.suffix

        # x = x.to(self._device)

        # audio = convert_audio(
        #     wav=x,
        #     sr=source_sr,
        #     target_sr=self.sample_rate,
        #     target_channels=self.num_channels,
        # # )
        # audio_batch = audio.unsqueeze(0)

        # with torch.no_grad():
        #     if self.output_file_format == "ecdc":
        #         file_like = io.BytesIO()
        #         create_ecdc(self.codec, audio, file_like, use_lm=False)
        #         file_like.seek(0)
        #         logging.info(f"Encoded ecdc with ENCODEC: {output_filename}")
        #         return [beam.pvalue.TaggedOutput('ecdc', (output_filename, file_like))]

        #     # The Encodec format is designed to be decoded live, so the channels
        #     # must be interleaved. Each "Frame" should be a tuple of (codebook, scale)
        #     frames = self.codec.encode(audio_batch)

        #     # From the docstring: "Each frame is a tuple `(codebook, scale)`, with
        #     # `codebook` of shape `[B, K, T]`, with `K` the number of codebooks."
        #     tensors = [t[0] for t in frames] # remove the "scale" from each frame
        #     codes = torch.cat(tensors, dim=2) # shape: [B, K, T]
        #     codes = codes.detach().cpu().numpy()

        # unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        # logging.info(f"Encoded with ENCODEC ({unbatched.shape}): {output_filename}")
        # return [(output_filename, unbatched)]
    
    # def process(self, src: tuple):
    #     song, elements = src
    #     return [(song, [item for element in elements for item in self.process_track(element)])]

class ExtractDAC(beam.DoFn):
    """Beam DoFn for extracting DAC tokens from audio."""

    def __init__(
        self,
        sample_rate: int,
        device: Optional[torch.device] = None,
    ):
        assert sample_rate in [
            16000,
            24000,
            44100,
        ], f"Invalid sample_rate: {sample_rate} for ExtractDAC"

        self.sample_rate = sample_rate
        self._device = device

    def setup(self):
        if self._device is None:
            self._device = get_device()

        self.codec = DAC()
        self.codec = load_model(
            tag="latest", model_type=SAMPLE_RATE_MAP[self.sample_rate]
        )
        self.codec.eval()
        self.codec.to(self._device)

    @property
    def suffix(self) -> str:
        return f".dac_{SAMPLE_RATE_MAP[self.sample_rate]}.npy"

    @property
    def num_channels(self) -> int:
        return 1

    def process(self, element: Tuple[str, torch.Tensor, int]):
        key, x, source_sr = element

        # Ensure that we are naming the file correctly.
        output_filename = remove_suffix(key, ".wav")
        output_filename = remove_suffix(output_filename, ".mp3")
        output_filename += self.suffix

        x = x.to(self._device)

        audio = convert_audio(
            wav=x,
            sr=source_sr,
            target_sr=self.sample_rate,
            target_channels=self.num_channels,
        )
        audio_batch = audio.unsqueeze(0)

        with torch.no_grad():
            audio_signal = AudioSignal(
                audio_batch, self.sample_rate, device=self._device
            )
            codes = encode(audio_signal, self._device, self.codec)["codes"]

        unbatched = codes.squeeze(0)  # `unbatched` has shape `[K, T]`
        logging.info(f"Encoded with DAC ({unbatched.shape}): {output_filename}")
        return [(output_filename, unbatched)]
