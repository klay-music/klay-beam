import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
import mock

from job_shard.transforms import (
    WriteManifest,
    ReadManifestFiles,
    ExtractVideoIdFromFile,
    FilterOutExistingFiles,
    ShardCopy,
    Enumerate,
)


class TestWriteManifest:
    def setup_method(self):
        self.dest_dir = "/tmp/test_dest/"
        self.audio_suffix = ".source.webm"
        self.transform = WriteManifest(
            dest_dir=self.dest_dir, audio_suffix=self.audio_suffix, min_shard_idx=0
        )

    def test_extract_video_id_from_path(self):
        """Test video ID extraction from file path."""
        test_cases = [
            ("/path/to/video123/video123.source.webm", "video123"),
            ("/bucket/shard-00001/abc456/abc456.source.webm", "abc456"),
            ("/deep/nested/path/xyz789/xyz789.source.webm", "xyz789"),
        ]

        for file_path, expected_video_id in test_cases:
            result = self.transform._extract_video_id(file_path)
            assert result == expected_video_id

    def test_extract_video_id_fallback(self):
        """Test fallback video ID extraction from filename."""
        file_path = "/video123.source.webm"  # Single level path to trigger fallback
        result = self.transform._extract_video_id(file_path)
        assert result == "video123"

    @mock.patch("job_shard.transforms.FileSystems.create")
    def test_process_writes_manifest(self, mock_create):
        """Test that process method writes MANIFEST.txt correctly."""
        mock_file = mock.Mock()
        mock_create.return_value.__enter__.return_value = mock_file

        # Create mock file metadata objects
        files = [
            mock.Mock(path="/src/video1/video1.source.webm"),
            mock.Mock(path="/src/video2/video2.source.webm"),
        ]

        element = (0, files)  # shard_idx=0, files

        results = list(self.transform.process(element))

        # Verify manifest was written
        expected_path = "/tmp/test_dest/shard-00000/MANIFEST.txt"
        mock_create.assert_called_once_with(expected_path)

        # Verify content written
        written_content = mock_file.write.call_args[0][0].decode("utf-8")
        assert "video1" in written_content
        assert "video2" in written_content

        # Verify element is yielded back
        assert results[0] == element


class TestReadManifestFiles:
    def setup_method(self):
        self.transform = ReadManifestFiles()

    @mock.patch("job_shard.transforms.FileSystems.open")
    def test_process_reads_manifest(self, mock_open_fs):
        """Test reading MANIFEST.txt file."""
        manifest_content = "video1\nvideo2\nvideo3\n"
        mock_file = mock.Mock()
        mock_file.read.return_value = manifest_content.encode("utf-8")
        mock_open_fs.return_value.__enter__.return_value = mock_file

        manifest_path = mock.Mock(path="/dest/shard-00000/MANIFEST.txt")

        results = list(self.transform.process(manifest_path))

        expected_video_ids = ["video1", "video2", "video3"]
        assert results == expected_video_ids

    @mock.patch("job_shard.transforms.FileSystems.open")
    def test_process_handles_empty_lines(self, mock_open_fs):
        """Test handling of empty lines in MANIFEST.txt."""
        manifest_content = "video1\n\nvideo2\n  \nvideo3\n"
        mock_file = mock.Mock()
        mock_file.read.return_value = manifest_content.encode("utf-8")
        mock_open_fs.return_value.__enter__.return_value = mock_file

        manifest_path = mock.Mock(path="/dest/shard-00000/MANIFEST.txt")

        results = list(self.transform.process(manifest_path))

        expected_video_ids = ["video1", "video2", "video3"]
        assert results == expected_video_ids


class TestExtractVideoIdFromFile:
    def setup_method(self):
        self.audio_suffix = ".source.webm"
        self.transform = ExtractVideoIdFromFile(self.audio_suffix)

    def test_extract_video_id_from_path_structure(self):
        """Test video ID extraction from directory structure."""
        test_cases = [
            ("/bucket/ytdl-data/video123/video123.source.webm", "video123"),
            ("/path/to/abc456/abc456.source.webm", "abc456"),
            ("/deep/nested/xyz789/xyz789.source.webm", "xyz789"),
        ]

        for file_path, expected_video_id in test_cases:
            result = self.transform._extract_video_id_from_path(file_path)
            assert result == expected_video_id

    def test_extract_video_id_fallback(self):
        """Test fallback to filename extraction."""
        file_path = "/video123.source.webm"  # Single level to trigger fallback
        result = self.transform._extract_video_id_from_path(file_path)
        assert result == "video123"

    def test_process(self):
        """Test the process method."""
        file_metadata = mock.Mock(
            path="/bucket/ytdl-data/video123/video123.source.webm"
        )

        results = list(self.transform.process(file_metadata))

        expected = [("video123", file_metadata)]
        assert results == expected


class TestFilterOutExistingFiles:
    def setup_method(self):
        self.transform = FilterOutExistingFiles()

    def test_process_no_existing_files(self):
        """Test processing when no existing files found."""
        file1 = mock.Mock(path="/src/video1/video1.source.webm")
        file2 = mock.Mock(path="/src/video2/video2.source.webm")

        # CoGroupByKey result: (video_id, {"files": [...], "existing": [...]})
        element = ("video1", {"files": [file1, file2], "existing": []})

        results = list(self.transform.process(element))

        assert results == [file1, file2]

    def test_process_with_existing_files(self):
        """Test processing when existing files found."""
        file1 = mock.Mock(path="/src/video1/video1.source.webm")

        # CoGroupByKey result: video ID exists in manifest
        element = ("video1", {"files": [file1], "existing": [True]})

        results = list(self.transform.process(element))

        # Should filter out (no results)
        assert results == []


class TestShardCopy:
    def setup_method(self):
        self.transform = ShardCopy(
            src_dir="/src/",
            dest_dir="/dest/",
            audio_suffix=".source.webm",
            suffixes=[".source.json", ".source.webm"],
            min_shard_idx=0,
        )

    def test_dst_path_generation(self):
        """Test destination path generation with 5-digit shard format."""
        from pathlib import Path

        rel_path = Path("video1/video1.json")
        shard_idx = 0

        result = self.transform._dst_path(rel_path, shard_idx)

        expected = "/dest/shard-00000/video1/video1.source.json"
        assert result == expected

    def test_dst_path_with_min_shard_idx(self):
        """Test destination path with non-zero min_shard_idx."""
        transform = ShardCopy(
            src_dir="/src/",
            dest_dir="/dest/",
            audio_suffix=".source.webm",
            suffixes=[".source.json"],
            min_shard_idx=5,
        )
        from pathlib import Path

        rel_path = Path("video1/video1.json")
        shard_idx = 0  # Will be adjusted to 5

        result = transform._dst_path(rel_path, shard_idx)

        expected = "/dest/shard-00005/video1/video1.source.json"
        assert result == expected


class TestEnumerate:
    def test_enumerate_transform(self):
        """Test the Enumerate PTransform."""
        with TestPipeline() as p:
            input_data = ["a", "b", "c"]

            result = p | beam.Create(input_data) | Enumerate(start=0)

            expected = [(0, "a"), (1, "b"), (2, "c")]
            assert_that(result, equal_to(expected))

    def test_enumerate_with_custom_start(self):
        """Test Enumerate with custom start value."""
        with TestPipeline() as p:
            input_data = ["x", "y"]

            result = p | beam.Create(input_data) | Enumerate(start=10)

            expected = [(10, "x"), (11, "y")]
            assert_that(result, equal_to(expected))
