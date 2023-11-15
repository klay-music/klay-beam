"""
This script is used to validate and evaluate DAC token arrays
written to file. Currently we only support loading files from disk.
"""

from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import scipy
import torch
from tqdm import trange

from dac.utils import load_model
from dac.model import DAC

from klay_beam.utils import get_device


SR = 44100
CHUNK_LENGTH = 100


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--chunk-length", type=int, default=CHUNK_LENGTH)
    parser.add_argument("--model-sample-rate", type=int, default=44100)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    device = get_device()

    # load DAC model
    dac = DAC()
    dac = load_model(tag="latest", model_type="44khz")
    dac.eval()
    dac.to(device)

    for fp in args.source_dir.glob("**/*.dac_44khz.npy"):
        print(f"Decoding codes from {fp}")

        # check if decoded audio already exists
        save_path = fp.with_suffix(".wav")
        if save_path.is_file():
            print(f"Decode audio file already exists at: {save_path}")
            continue

        # load codes from file
        codes = np.load(fp).astype(np.int64)
        codes = torch.from_numpy(codes).to(torch.int64).unsqueeze(0)

        # chunked reconstruction loop
        recons = []
        for i in trange(0, codes.shape[-1], args.chunk_length):
            c = codes[..., i : i + args.chunk_length].to(device)
            z = dac.quantizer.from_codes(c)[0]
            r = dac.decode(z)
            recons.append(r["audio"].detach().cpu().squeeze(0))

        # write audio to file
        audio = torch.cat(recons, dim=-1).numpy().squeeze(0)
        scipy.io.wavfile.write(save_path, SR, audio)
