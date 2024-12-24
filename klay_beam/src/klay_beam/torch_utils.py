from typing import Tuple, Optional
import logging


def torch_check() -> Tuple[bool, Optional[ImportError]]:
    torch_available, torch_import_error = False, None
    try:
        import torch  # noqa
        import torchaudio  # noqa

        torch_available = True
    except ImportError as e:
        torch_import_error = e
        logging.info(f"torch is not available: {e}")
    return torch_available, torch_import_error


TORCH_AVAILABLE, TORCH_IMPORT_ERROR = torch_check()


def ensure_torch_available():
    if not TORCH_AVAILABLE:
        raise ImportError(
            f"This features requires a compatible torch version. {TORCH_IMPORT_ERROR}"
        )
