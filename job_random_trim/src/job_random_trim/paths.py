from pathlib import PurePosixPath
import os.path


def get_target_path(input_uri: str, source_dir: str, target_dir: str) -> str:
    """
    When processing datasets, we often have multi-layer directories, and we need to
    map input files to output files. For example, inputs could look like this:

    ```
    gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3
    gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1012000.mp3
    gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009601.mp3
    gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009701.mp3
    ```

    And we want to transform them to outputs like this:
    ```
    gs://some-other-bucket/jamendo/00/1009600.source.wav
    gs://some-other-bucket/jamendo/00/1012000.source.wav
    gs://some-other-bucket/jamendo/01/1009601.source.wav
    gs://some-other-bucket/jamendo/01/1009701.source.wav
    ```

    Given an source directory and a target directory, map the input filenames to the
    output filenames, preserving the relative directory structure. This should work
    across local paths and GCS URIs.
    """
    relative_source_filename = PurePosixPath(input_uri).relative_to(source_dir)
    relative_target_filename = relative_source_filename.with_suffix(".source.wav")

    # pathlib does not safely handle `//` in URIs
    # `assert str(pathlib.Path("gs://data")) == "gs://data"` fails
    #
    # As a result, we just use os.path.join. Mixing pathlib and os.path is suboptimal.
    # Is there a better way?
    return os.path.join(target_dir, relative_target_filename)


assert os.path.sep == "/", "os.path.join (in get_target_path) breaks on Windows"
