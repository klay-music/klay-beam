import logging
from functools import wraps

TORCH_AVAILABLE = False
TORCH_IMPORT_ERROR = None

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError as e:
    TORCH_IMPORT_ERROR = e
    logging.info(f"torch is not available: {e}")


def skip_if_no_torch(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TORCH_AVAILABLE:
            logging.warning(
                f"Skipping {func.__name__} because torch is not available: "
                f"{TORCH_IMPORT_ERROR}"
            )
            return None
        return func(*args, **kwargs)

    return wrapper


def import_torch():
    if TORCH_AVAILABLE:
        return torch
    else:
        return None
