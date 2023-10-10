from pathlib import PurePosixPath
from urllib.parse import urlparse, urlunparse
import os.path
import apache_beam as beam


def move(filename, source_dir, target_dir):
    """Move a file from source_dir to target_dir, preserving the relative
    directory structure. Should work across local paths and GCS URIs. Throw if
    input is not in source_dir.

    ```
    # Example usage:
    move(
        'gs://klay-datasets/audios/00/1009600.mp3',
        'gs://klay-datasets/audios/',
        '/somewhere/else/'
    ) == '/somewhere/else/00/1009600.mp3'
    ```
    """
    relative_source_filename = PurePosixPath(filename).relative_to(source_dir)
    return os.path.join(target_dir, relative_source_filename)

    # If we wanted to avoid os.path.join, we could detect URIs and handle them
    # separately. However, this is more complicated and also uses the _replace
    # method (which is technically private). See:
    # https://stackoverflow.com/questions/38552253/change-urlparse-path-of-a-url

    result = str(PurePosixPath(target_dir) / relative_source_filename)
    target_dir_is_uri = urlparse(target_dir).scheme != ""
    if target_dir_is_uri:
        target_uri = urlparse(target_dir)
        new_path = str(
            PurePosixPath(target_uri.path) / PurePosixPath(relative_source_filename)
        )
        target_uri = target_uri._replace(path=new_path)
        result = urlunparse(target_uri)

    return result


assert os.path.sep == "/", "os.path.join (in get_target_path) breaks on Windows"


def remove_suffix(path: str, suffix: str):
    if path.endswith(suffix):
        return path[: -len(suffix)]
    return path
