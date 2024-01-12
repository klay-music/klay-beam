from apache_beam.io.filesystem import FileMetadata
from pathlib import Path
from klay_beam.transforms import SkipCompleted


def test_skip_completed():
    this_file = Path(__file__)
    data_dir = str(this_file.parent / "test_data")

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
    assert does_not_skip.process(metadata_02) == [
        metadata_02
    ], "SkipCompleted should return the input file when some target files are missing"

    # test overwrite
    metadata_03 = FileMetadata(
        path=str(Path(data_dir) / "exists.wav"),
        size_in_bytes=16,
    )
    does_not_skip = SkipCompleted(
        old_suffix=".wav",
        new_suffix=".01.txt",
        overwrite=True,
    )
    assert does_not_skip.process(metadata_03) == [
        metadata_03
    ], "SkipCompleted should return the input file when overwrite is True"


def test_skip_completed_with_timestamp():
    this_file = Path(__file__)
    data_dir = str(this_file.parent / "test_data")
    does_not_skip = SkipCompleted(
        old_suffix=".wav",
        new_suffix=".01.txt",
        check_timestamp=True,
    )
    source1 = FileMetadata(
        path=str(Path(data_dir) / "exists.wav"),
        size_in_bytes=16,
        last_updated_in_seconds=float("inf"),
    )
    assert does_not_skip.process(source1) == [source1], (
        "SkipCompleted with `check_timestamp=True`should return the input file when the target "
        "file is present but older than the source file"
    )

    source2 = FileMetadata(
        path=str(Path(data_dir) / "exists.wav"),
        size_in_bytes=16,
        last_updated_in_seconds=1,
    )
    assert does_not_skip.process(source2) == [], (
        "SkipCompleted with `check_timestamp=True`should return an empty list when the target "
        "file  is present and newer than the source file"
    )
