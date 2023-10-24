import logging
from functools import wraps

from klay_beam.torch_utils import TORCH_AVAILABLE, TORCH_IMPORT_ERROR


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
