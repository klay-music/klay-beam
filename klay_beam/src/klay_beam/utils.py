import torch
from pkg_resources import parse_version


def get_device(mps_valid: bool = False) -> torch.device:
    # torch.has_mps is not available in older versions of torch such as 1.11.
    # Ensure it exists before using it. This also satisfies mypy.
    if mps_valid and hasattr(torch, "has_mps") and torch.has_mps:
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
