import tempfile
import os
import mock
import shutil
import apache_beam as beam
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that, equal_to
from apache_beam.io import fileio
import pytest

from job_shard.transforms import (
    LoadExistingVideoIds,
    ExtractVideoIdFromFile,
    FilterOutExistingFiles,
    WriteManifest,
)


class SimpleFileMetadata:
    """Simple file metadata class for testing (can be pickled)."""

    def __init__(self, path):
        self.path = path

    def __eq__(self, other):
        if isinstance(other, SimpleFileMetadata):
            return self.path == other.path
        return False

    def __repr__(self):
        return f"SimpleFileMetadata('{self.path}')"


class TestPipelineIntegration:
    def setup_method(self):
        self.test_dir = tempfile.mkdtemp()

    def teardown_method(self):
        shutil.rmtree(self.test_dir)

    def test_load_existing_video_ids_empty(self):
        """Test LoadExistingVideoIds when no MANIFEST files exist."""
        dest_dir = os.path.join(self.test_dir, "dest")
        os.makedirs(dest_dir, exist_ok=True)

        with TestPipeline() as p:
            # Create a dummy input to trigger the pipeline
            dummy = p | beam.Create([1])

            result = dummy | LoadExistingVideoIds(dest_dir=dest_dir)

            # Should be empty when no MANIFEST files exist
            assert_that(result, equal_to([]))

    def test_load_existing_video_ids_with_manifests(self):
        """Test LoadExistingVideoIds with existing MANIFEST files."""
        dest_dir = os.path.join(self.test_dir, "dest")

        # Create shard directories with MANIFEST files
        shard1_dir = os.path.join(dest_dir, "shard-00000")
        shard2_dir = os.path.join(dest_dir, "shard-00001")
        os.makedirs(shard1_dir, exist_ok=True)
        os.makedirs(shard2_dir, exist_ok=True)

        # Write MANIFEST files
        with open(os.path.join(shard1_dir, "MANIFEST.txt"), "w") as f:
            f.write("video1\nvideo2\n")

        with open(os.path.join(shard2_dir, "MANIFEST.txt"), "w") as f:
            f.write("video3\nvideo4\n")

        with TestPipeline() as p:
            dummy = p | beam.Create([1])

            result = dummy | LoadExistingVideoIds(dest_dir=dest_dir)

            expected = [
                ("video1", True),
                ("video2", True),
                ("video3", True),
                ("video4", True),
            ]
            assert_that(result, equal_to(expected))

    def test_extract_and_filter_pipeline(self):
        """Test the complete extract and filter pipeline."""
        # Simple file metadata objects that can be pickled
        file1 = SimpleFileMetadata("/src/ytdl-data/video1/video1.source.webm")
        file2 = SimpleFileMetadata("/src/ytdl-data/video2/video2.source.webm")
        file3 = SimpleFileMetadata("/src/ytdl-data/video3/video3.source.webm")

        with TestPipeline() as p:
            # Create input files
            files = p | "CreateFiles" >> beam.Create([file1, file2, file3])

            # Create existing video IDs (video2 already exists)
            existing = p | "CreateExisting" >> beam.Create([("video2", True)])

            # Extract video IDs from files
            keyed_files = files | "KeyFiles" >> beam.ParDo(
                ExtractVideoIdFromFile(audio_suffix=".source.webm")
            )

            # Join and filter
            result = (
                {"files": keyed_files, "existing": existing}
                | "CoGroup" >> beam.CoGroupByKey()
                | "Filter" >> beam.ParDo(FilterOutExistingFiles())
            )

            # Should only return file1 and file3 (video2 is filtered out)
            expected = [file1, file3]
            assert_that(result, equal_to(expected))

    def test_write_manifest_pipeline(self):
        """Test the WriteManifest transform in a pipeline context."""
        dest_dir = os.path.join(self.test_dir, "dest")
        os.makedirs(dest_dir, exist_ok=True)

        # Simple file metadata objects that can be pickled
        file1 = SimpleFileMetadata("/src/ytdl-data/video1/video1.source.webm")
        file2 = SimpleFileMetadata("/src/ytdl-data/video2/video2.source.webm")

        with TestPipeline() as p:
            # Create shard data (shard_idx, files)
            shard_data = p | beam.Create([(0, [file1, file2])])

            with mock.patch("job_shard.transforms.FileSystems.create") as mock_create:
                mock_file = mock.Mock()
                mock_create.return_value.__enter__.return_value = mock_file

                result = shard_data | "WriteManifest" >> beam.ParDo(
                    WriteManifest(
                        dest_dir=dest_dir, audio_suffix=".source.webm", min_shard_idx=0
                    )
                )

                # The transform should yield the input back
                assert_that(result, equal_to([(0, [file1, file2])]))

    def test_complete_filtering_workflow(self):
        """Test the complete workflow: load existing, extract, filter."""
        dest_dir = os.path.join(self.test_dir, "dest")

        # Create shard directory with MANIFEST file
        shard_dir = os.path.join(dest_dir, "shard-00000")
        os.makedirs(shard_dir, exist_ok=True)

        with open(os.path.join(shard_dir, "MANIFEST.txt"), "w") as f:
            f.write("video2\nvideo4\n")

        # Simple file metadata objects that can be pickled
        file1 = SimpleFileMetadata("/src/ytdl-data/video1/video1.source.webm")
        file2 = SimpleFileMetadata("/src/ytdl-data/video2/video2.source.webm")
        file3 = SimpleFileMetadata("/src/ytdl-data/video3/video3.source.webm")
        file4 = SimpleFileMetadata("/src/ytdl-data/video4/video4.source.webm")

        with TestPipeline() as p:
            # Input files
            files = p | "CreateFiles" >> beam.Create([file1, file2, file3, file4])

            # Load existing video IDs
            existing_video_ids = files | LoadExistingVideoIds(dest_dir=dest_dir)

            # Key files by video ID
            keyed_files = files | "KeyFiles" >> beam.ParDo(
                ExtractVideoIdFromFile(audio_suffix=".source.webm")
            )

            # Join and filter
            filtered_files = (
                {"files": keyed_files, "existing": existing_video_ids}
                | "CoGroup" >> beam.CoGroupByKey()
                | "Filter" >> beam.ParDo(FilterOutExistingFiles())
            )

            # Should only return file1 and file3 (video2 and video4 filtered out)
            expected = [file1, file3]
            assert_that(filtered_files, equal_to(expected))


class TestPipelineEdgeCases:
    def test_extract_video_id_edge_cases(self):
        """Test video ID extraction edge cases."""
        transform = ExtractVideoIdFromFile(audio_suffix=".source.webm")

        test_cases = [
            # Normal case
            ("/bucket/ytdl-data/abc123/abc123.source.webm", "abc123"),
            # Deep nesting
            ("/very/deep/nested/path/xyz789/xyz789.source.webm", "xyz789"),
            # Short path (fallback to filename)
            ("video123.source.webm", "video123"),
            # Different audio suffix
            ("/path/video456/video456.source.mp4", "video456"),
        ]

        for path, expected in test_cases:
            file_meta = SimpleFileMetadata(path)
            result = list(transform.process(file_meta))
            assert result[0][0] == expected  # video_id part
            assert result[0][1] == file_meta  # file_metadata part

    def test_filter_empty_groups(self):
        """Test FilterOutExistingFiles with empty groups."""
        transform = FilterOutExistingFiles()

        # Test empty files list
        result1 = list(transform.process(("video1", {"files": [], "existing": []})))
        assert result1 == []

        # Test empty existing list
        file1 = SimpleFileMetadata("/src/video1/video1.source.webm")
        result2 = list(
            transform.process(("video1", {"files": [file1], "existing": []}))
        )
        assert result2 == [file1]
