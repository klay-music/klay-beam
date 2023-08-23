import io
import logging
import note_seq
from note_seq import midi_io
import numpy as np
import scipy
from typing import Optional, Tuple, List


def write_midi_file(note_sequence_tuple) -> None:
    fname, note_sequence = note_sequence_tuple
    if note_sequence is None:
        logging.error(f"Note sequence is None for {fname}. Skipping.")
        return

    out_filename = add_suffix(fname, ".mid")
    midi_io.sequence_proto_to_midi_file(note_sequence, out_filename)


def array_to_bytes(
    audio_tuple: Tuple[str, np.ndarray, int]
) -> List[Tuple[str, bytes, int]]:
    fname, audio, sr = audio_tuple

    buf = io.BytesIO()
    scipy.io.wavfile.write(buf, sr, audio)
    buf.seek(0)
    wav_data = buf.read()

    return [(fname, wav_data, sr)]


def remove_suffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path[: -len(suffix)]
    return path


def add_suffix(path: str, suffix: str):
    assert suffix.startswith(".")
    while path.endswith("."):
        path = path[:-1]

    if path.endswith(suffix):
        return path
    else:
        return path + suffix


