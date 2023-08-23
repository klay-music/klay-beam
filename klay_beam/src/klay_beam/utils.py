import torch
from pkg_resources import parse_version


def get_device(mps_valid: bool = False) -> torch.device:
    if mps_valid and torch.has_mps:
        return torch.device("mps")
    elif torch.cuda.is_available():
        # if cuda version < 11, RTX 30xx series not available
        if parse_version(torch.version.cuda) < parse_version("11"):  # type: ignore
            for i in range(torch.cuda.device_count()):
                if "RTX 30" not in torch.cuda.get_device_name(i):
                    return torch.device("cuda", i)
        else:
            return torch.device("cuda")
    return torch.device("cpu")
