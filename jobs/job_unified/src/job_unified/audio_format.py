"""Audio format handling for the unified processor."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

import torch
import torchaudio


class AudioFormat(str, Enum):
    """Supported audio formats."""
    OPUS = "opus"
    VORBIS = "ogg"
    MP3 = "mp3"
    WAV = "wav"


@dataclass
class FormatConfig:
    """Configuration for an audio format."""
    extension: str
    format: Optional[str]  # Format string for torchaudio.save
    compression: Optional[float] = None  # For formats that support compression level
    bits_per_sample: Optional[int] = None  # For WAV
    encoding: Optional[str] = None  # Specific encoding format
    bitrate: Optional[int] = None  # For lossy formats


FORMAT_CONFIGS = {
    AudioFormat.OPUS: FormatConfig(
        extension=".opus",
        format="opus",
        compression=10,  # 0-10, higher is better quality
        bitrate=128000,  # 128kbps
    ),
    AudioFormat.VORBIS: FormatConfig(
        extension=".ogg",
        format="vorbis",
        compression=5,  # 0-10, higher is better quality
    ),
    AudioFormat.MP3: FormatConfig(
        extension=".mp3",
        format="mp3",
        bitrate=320000,  # 320kbps
    ),
    AudioFormat.WAV: FormatConfig(
        extension=".wav",
        format=None,  # Use default WAV format
        bits_per_sample=16,
    ),
}


def save_audio(
    tensor: torch.Tensor,
    path: str | Path,
    sample_rate: int,
    format_: AudioFormat,
    normalize: bool = True,
) -> None:
    """Save audio tensor to file in specified format.
    
    Args:
        tensor: Audio tensor to save (channels, samples)
        path: Output path (extension will be added if not present)
        sample_rate: Sample rate of the audio
        format_: Output format to use
        normalize: Whether to normalize the audio before saving
    """
    path = Path(path)
    config = FORMAT_CONFIGS[format_]
    
    # Add extension if not present
    if not str(path).endswith(config.extension):
        path = path.with_suffix(config.extension)

    # Normalize if requested
    if normalize:
        tensor = tensor / torch.max(torch.abs(tensor))

    # Prepare save arguments
    kwargs = {}
    if config.format is not None:
        kwargs["format"] = config.format
    # Remove compression for now due to torchaudio API change
    # if config.compression is not None:
    #     kwargs["compression"] = config.compression
    if config.bits_per_sample is not None:
        kwargs["bits_per_sample"] = config.bits_per_sample
    if config.encoding is not None:
        kwargs["encoding"] = config.encoding
    
    # Save with format-specific settings
    torchaudio.save(str(path), tensor, sample_rate, **kwargs)


def get_audio_suffix(format_: AudioFormat) -> str:
    """Get the file extension for a given format."""
    return FORMAT_CONFIGS[format_].extension 