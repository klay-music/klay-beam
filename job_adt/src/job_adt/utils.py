import io
import logging
import note_seq
from note_seq import midi_io
import numpy as np
import scipy
from typing import Optional, Tuple, List
from apache_beam.io.filesystems import FileSystems


def note_sequence_to_midi(note_sequence_tuple):
    fname, note_sequence = note_sequence_tuple
    if note_sequence is None:
        logging.error(f"Note sequence is None for {fname}. Skipping.")
        return []

    out_filename = add_suffix(fname, ".mid")

    # Pattern for writing in-memory files is copied from the note_seq library:
    # https://github.com/magenta/note-seq/blob/5b657f8b29c9fbd3d72d6a581f9abe8bc0b90c53/note_seq/midi_io.py#L190-L207
    pretty_midi_object = midi_io.note_sequence_to_pretty_midi(note_sequence)
    file_like = io.BytesIO()
    pretty_midi_object.write(file_like)
    file_like.seek(0)

    return [(out_filename, file_like)]


# Copied from klay_beam
def write_file(output_path_and_buffer):
    """Helper function for writing a buffer to a given path. This should be able
    to handle gs:// style paths as well as local paths.

    Can be used with beam.Map(write_file)
    """
    output_path, buffer = output_path_and_buffer
    logging.info("Writing to: {}".format(output_path))
    with FileSystems.create(output_path) as file_handle:
        file_handle.write(buffer.read())

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
