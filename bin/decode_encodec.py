"""
This script is used to validate and evaluate EnCodec token arrays saved as .npy
files. Note that as of September 2023, we save mono (24kHz) encodec tokens as
.npy files and stereo (48kHz) encodec tokens as .ecdc files. ecdc files can be
decoded with the `python -m encodec` tool.

Currently we only support loading files from disk.
"""

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import scipy
import torch
from tqdm import trange


from encodec import EncodecModel
import torchaudio

from klay_beam.utils import get_device


SR = 44100
CHUNK_LENGTH = 100


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--chunk-length", type=int, default=CHUNK_LENGTH)
    parser.add_argument("--model-sample-rate", type=int, default=48000, choices=[24000, 48000])
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    # load DAC model
    encodec: EncodecModel
    glob: str
    if args.model_sample_rate == 24000:
        encodec = EncodecModel.encodec_model_24khz()
        glob = "**/*.encodec_24khz.npy"
    elif args.model_sample_rate == 48000:
        raise NotImplementedError("48kHz encodec model not yet supported")
        encodec = EncodecModel.encodec_model_48khz()
        encodec.set_target_bandwidth(24)
        glob = "**/*.encodec_48khz.npy"

    encodec.eval()
    encodec.to(device)

    for fp in args.source_dir.glob(glob):
        print(f"Decoding codes from {fp}")

        # check if decoded audio already exists
        save_path = fp.with_suffix(".wav")
        if save_path.is_file():
            print(f"Decode audio file already exists at: {save_path}")
            continue

        # load codes from file
        codes = np.load(fp).astype(np.int64)
        codes = torch.from_numpy(codes).to(torch.int64) # [K, T]

        print(f"Loaded codes from: {fp} shape: {codes.shape}")

        batched_codes = codes.unsqueeze(0) # [1, K, T]
        frames = [(batched_codes, None)]

        # batched_codes = codes.unsqueeze(0)
        # frames = [(batched_codes, None)]
        batched_audio = encodec.decode(frames).detach()

        audio = batched_audio[0] # (-1, 1)
        torchaudio.save(save_path, audio, args.model_sample_rate)

        print(f"Decoded audio with shape: {audio.shape} min: {audio.min()} max: {audio.max()}")
