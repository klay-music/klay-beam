import numpy as np
import torch

from job_demucs.demucs import DemucsSeparator


def test_demucs_separator():
    sr = 44100
    signal = torch.rand((2, sr), dtype=torch.float32)
    separator = DemucsSeparator(model_name="mdx_q")
    output = separator(signal)
    for k, s in output.items():
        assert k in ["vocals", "other", "drums", "bass"]
        assert type(s) == np.ndarray
        assert len(s) == len(signal)
