from typing import Tuple, Optional
import logging

def torch_check() -> Tuple[bool, Optional[str]]:
    torch_available, torch_import_error = False, None
    try:
        import torch
        import torchaudio

        if torchaudio.__version__ < "0.8.0":
            raise ImportError(
                "Incompatible version of torchaudio is installed. Install version 0.8.0 or newer."
            )
    
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
