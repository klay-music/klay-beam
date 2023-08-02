import demucs.apply as apply
import demucs.pretrained as pretrained
import numpy as np
import torch
from typing import Optional


class DemucsSeparator:
    sr = 44100

    def __init__(
        self,
        model_name: str = "mdx_extra",
        num_workers: int = 1,
        overlap: float = 0.25,
        device: Optional[torch.device] = None,
        shifts: bool = True,
        split: bool = True,
        progress_bar: bool = False,
    ):
        """Separates a signal into it's sources using demucs.

        Arguments
        ---------
        model_name
            Pretrained model name. The list of pre-trained models is:
                `mdx`: trained only on MusDB HQ, winning model on track A
                       at the MDX challenge.
                `mdx_extra`: trained with extra training data (including
                             MusDB test set). ranked 2nd on the track B of
                             the MDX challenge.
                `mdx_q`, `mdx_extra_q`: quantized version of the previous models.
                                        Smaller download and storage but quality
                                        can be slightly worse.
        num_workers
            How many workers to run the separation in parallel.
        overlap
            Overlap between splits
        device
            Which device should we use to run the computation.
        shifts
            if > 0, will shift in time `mix` by a random amount between 0 and 0.5 sec and apply the
            oppositve shift to the output. This is repeated `shifts` time and all predictions are
            averaged. This effectively makes the model time equivariant and improves SDR by up to
            0.2 points.
        split
            if True, the input will be broken down in 8 seconds extracts and predictions will be
            performed individually on each and concatenated. Useful for model with large memory
            footprint like Tasnet.
        """
        self.model = pretrained.get_model(model_name)
        self.num_workers = num_workers
        self.overlap = overlap
        self.device = device or torch.device("cpu")
        self.shifts = shifts
        self.split = split
        self.progress_bar = progress_bar

    def __call__(self, signal: torch.Tensor):
        # normalize
        ref = signal.mean(0)
        signal = (signal - signal.mean()) / signal.std()

        # separate
        sources = apply.apply_model(
            self.model,
            signal[None],
            device=self.device,
            shifts=self.shifts,
            split=self.split,
            overlap=self.overlap,
            progress=self.progress_bar,
            num_workers=self.num_workers,
        )[0]
        sources = sources * ref.std() + ref.mean()

        return self._construct_dict(list(sources), self.model.sources)

    @staticmethod
    def _construct_dict(
        sources: list[torch.Tensor], sources_names: list[str]
    ) -> dict[str, np.ndarray]:
        d = {}
        for name, signal in zip(sources_names, sources):
            d[name] = signal.cpu().detach().numpy()
        return d
