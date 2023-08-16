from pathlib import Path
from apache_beam.io.filesystem import FileMetadata
from klay_beam.transforms import SkipCompleted


def test_skip_completed():
    this_file = Path(__file__)
    data_dir = str(this_file.parent / "data")


    skips = SkipCompleted(
        old_suffix=".wav",
        new_suffix=".01.txt",
    )
    metadata_01 = FileMetadata(
        path=str(Path(data_dir) / "exists.wav"),
        size_in_bytes=16,
    )
    assert (
        skips.process(metadata_01) == []
    ), "SkipCompleted should return and empty list when the target file is present"

    metadata_02 = FileMetadata(
        path="/dummy/dir/exists.wav",
        size_in_bytes=16,
    )
    skips = SkipCompleted(
        old_suffix=".wav",
        new_suffix=[".01.txt", ".02.txt", ".03.txt"],
        source_dir="/dummy/dir",
        target_dir=data_dir,
    )
    assert (
        skips.process(metadata_02) == []
    ), "SkipCompleted should return and empty list when all the target files are present"


    does_not_skip = SkipCompleted(
        old_suffix=".wav",
        new_suffix=[".01.txt", ".02.txt", ".03.txt", ".04.txt"],
        source_dir="/dummy/dir",
        target_dir=data_dir,
    )
    assert (
        does_not_skip.process(metadata_02) == [metadata_02]
    ), "SkipCompleted should return the input file when some target files are missing"