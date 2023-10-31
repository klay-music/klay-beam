import pathlib
import io
from typing import Optional, Union, List
import apache_beam as beam
from apache_beam.io import filesystems
import numpy as np
import logging

from apache_beam.io.filesystem import FileMetadata
from apache_beam.io.filesystems import FileSystems

from .path import move, remove_suffix


class SkipCompleted(beam.DoFn):
    def __init__(
        self,
        old_suffix: str,
        new_suffix: Union[str, List[str]],
        source_dir: Optional[str] = None,
        target_dir: Optional[str] = None,
        check_timestamp: bool = False,
    ):
        if isinstance(new_suffix, str):
            new_suffix = [new_suffix]
        self._new_suffixes = new_suffix
        self._old_suffix = old_suffix

        assert (source_dir is None) == (
            target_dir is None
        ), "source_dir and target_dir must both be None or strings"

        self._source_dir = source_dir
        self._target_dir = target_dir
        self._check_timestamp = check_timestamp

    def process(self, source_metadata: FileMetadata):
        check = remove_suffix(source_metadata.path, self._old_suffix)
        if self._source_dir is not None:
            check = move(check, self._source_dir, self._target_dir)
        checks = [check + suffix for suffix in self._new_suffixes]
        limits = [1 for _ in checks]

        results = FileSystems.match(checks, limits=limits)
        assert len(results) > 0, "Unexpected empty results. This should never happen."

        for result in results:
            num_matches = len(result.metadata_list)
            logging.info(f"Found {num_matches} of: {result.pattern}")
            if num_matches != 0 and self._check_timestamp:
                for target_metadata in result.metadata_list:
                    if (
                        target_metadata.last_updated_in_seconds
                        < source_metadata.last_updated_in_seconds
                    ):
                        logging.info(
                            f"Do not skip! A target was found ({target_metadata.path}), but it is "
                            f"older than source file ({source_metadata.path})"
                        )
                        return [source_metadata]
            elif num_matches == 0:
                return [source_metadata]

        logging.info(f"Targets already exist. Skipping: {source_metadata.path}")
        return []


def numpy_to_file(numpy_data: np.ndarray):
    in_memory_file_buffer = io.BytesIO()
    np.save(in_memory_file_buffer, numpy_data)
    in_memory_file_buffer.seek(0)
    return in_memory_file_buffer


def write_file(output_path_and_buffer):
    """Helper function for writing a buffer to a given path. This should be able
    to handle gs:// style paths as well as local paths.

    Can be used with beam.Map(write_file)
    """
    output_path, buffer = output_path_and_buffer
    logging.info("Writing to: {}".format(output_path))
    with filesystems.FileSystems.create(output_path) as file_handle:
        file_handle.write(buffer.read())
