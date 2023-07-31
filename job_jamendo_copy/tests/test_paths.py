from job_jamendo_copy.paths import get_target_path
import pathlib
import pytest


def test_get_target_path_basic():
    source_dir = "gs://klay-datasets/mtg_jamendo_autotagging/audios/"
    target_dir = "gs://klay-datasets/mtg_jamendo/"

    source_uris = [
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1012000.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009601.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009701.mp3",
    ]
    target_uris = [
        "gs://klay-datasets/mtg_jamendo/00/1009600.source.wav",
        "gs://klay-datasets/mtg_jamendo/00/1012000.source.wav",
        "gs://klay-datasets/mtg_jamendo/01/1009601.source.wav",
        "gs://klay-datasets/mtg_jamendo/01/1009701.source.wav",
    ]

    for source, expected in zip(source_uris, target_uris):
        actual_output_filename = get_target_path(source, source_dir, target_dir)
        assert actual_output_filename == expected


def test_get_target_path_errors():
    target_dir = "gs://klay-datasets/mtg_jamendo/"

    # When source file is not in the source_dir, raise an error
    erroneous_source_dir = "gs://klay-datasets/ELSEWHERE"
    source_uri = "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3"
    with pytest.raises(ValueError):
        get_target_path(source_uri, erroneous_source_dir, target_dir)

    # The following are different paths
    # 1. "/klay-datasets/"
    # 2. "gs://klay-datasets/"
    erroneous_source_dir = "/klay-datasets/mtg_jamendo_autotagging/audios/"
    source_uri = "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3"
    with pytest.raises(ValueError):
        get_target_path(source_uri, erroneous_source_dir, target_dir)


def test_get_target_path_gcp_to_local():
    source_dir = "gs://klay-datasets/mtg_jamendo_autotagging/audios/"
    target_dir = "/klay-datasets/mtg_jamendo"

    source_uris = [
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1012000.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009601.mp3",
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/01/1009701.mp3",
    ]
    target_uris = [
        "/klay-datasets/mtg_jamendo/00/1009600.source.wav",
        "/klay-datasets/mtg_jamendo/00/1012000.source.wav",
        "/klay-datasets/mtg_jamendo/01/1009601.source.wav",
        "/klay-datasets/mtg_jamendo/01/1009701.source.wav",
    ]

    for source, expected in zip(source_uris, target_uris):
        actual_output_filename = get_target_path(source, source_dir, target_dir)
        assert actual_output_filename == expected


def test_get_target_path_local_to_gcp():
    source_dir = "/klay-datasets/mtg_jamendo_autotagging/audios/"
    target_dir = "gs://klay-datasets/mtg_jamendo/"

    source_uris = [
        "/klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3",
        "/klay-datasets/mtg_jamendo_autotagging/audios/00/1012000.mp3",
        "/klay-datasets/mtg_jamendo_autotagging/audios/01/1009601.mp3",
        "/klay-datasets/mtg_jamendo_autotagging/audios/01/1009701.mp3",
    ]
    target_uris = [
        "gs://klay-datasets/mtg_jamendo/00/1009600.source.wav",
        "gs://klay-datasets/mtg_jamendo/00/1012000.source.wav",
        "gs://klay-datasets/mtg_jamendo/01/1009601.source.wav",
        "gs://klay-datasets/mtg_jamendo/01/1009701.source.wav",
    ]

    for source, expected in zip(source_uris, target_uris):
        actual_output_filename = get_target_path(source, source_dir, target_dir)
        assert actual_output_filename == expected


def test_pathlib_relative_to():
    # Ensure that pathlib.Path.relative_to can handle both paths and URIs
    uri = "gs://klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3"
    result1 = pathlib.PurePosixPath(uri).relative_to(
        "gs://klay-datasets/mtg_jamendo_autotagging/audios/"
    )
    assert result1 == pathlib.PurePosixPath("00/1009600.mp3")
    result2 = pathlib.PurePosixPath(uri).relative_to(
        "gs://klay-datasets/mtg_jamendo_autotagging/audios"
    )
    assert result2 == pathlib.PurePosixPath("00/1009600.mp3")

    filepath = "/klay-datasets/mtg_jamendo_autotagging/audios/00/1009600.mp3"
    result3 = pathlib.PurePosixPath(filepath).relative_to(
        "/klay-datasets/mtg_jamendo_autotagging/audios/"
    )
    assert result3 == pathlib.PurePosixPath("00/1009600.mp3")
    result4 = pathlib.PurePosixPath(filepath).relative_to(
        "/klay-datasets/mtg_jamendo_autotagging/audios"
    )
    assert result4 == pathlib.PurePosixPath("00/1009600.mp3")
