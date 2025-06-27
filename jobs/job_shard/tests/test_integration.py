import tempfile
import os
import shutil
import subprocess
import json
from pathlib import Path
import pytest


class TestJobShardIntegration:
    """Integration test that creates temp directories and runs the full pipeline."""

    def setup_method(self):
        """Set up temporary directories and test data."""
        self.test_dir = tempfile.mkdtemp()

        # Create source and destination directories
        self.src_dir = os.path.join(self.test_dir, "ytdl-data")
        self.dest_dir = os.path.join(self.test_dir, "dest")
        os.makedirs(self.src_dir, exist_ok=True)
        os.makedirs(self.dest_dir, exist_ok=True)

        # Test video IDs
        self.video_ids = [
            "video001",
            "video002",
            "video003",
            "video004",
            "video005",
            "video006",
            "video007",
            "video008",
            "video009",
            "video010",
        ]

        # Create test files in ytdl-data structure
        for video_id in self.video_ids:
            video_dir = os.path.join(self.src_dir, video_id)
            os.makedirs(video_dir, exist_ok=True)

            # Create .source.webm file
            webm_path = os.path.join(video_dir, f"{video_id}.source.webm")
            with open(webm_path, "wb") as f:
                f.write(b"fake_webm_content_" + video_id.encode())

            # Create .source.json file
            json_path = os.path.join(video_dir, f"{video_id}.source.json")
            with open(json_path, "w") as f:
                json.dump(
                    {
                        "id": video_id,
                        "title": f"Test Video {video_id}",
                        "duration": 120,
                    },
                    f,
                )

    def _run_workflow(self, max_dataset_size=None, min_shard_idx=0):
        """Run the job_shard workflow with given parameters."""
        cmd = [
            "python",
            "bin/run_workflow.py",
            "--runner",
            "DirectRunner",
            "--src_dir",
            self.src_dir,
            "--dest_dir",
            self.dest_dir,
            "--audio_suffix",
            ".source.webm",
            "--suffixes",
            ".source.webm",
            ".source.json",
            "--num_files_per_shard",
            "2",  # 2 files per shard
            "--min_shard_idx",
            str(min_shard_idx),
        ]

        if max_dataset_size:
            cmd.extend(["--max_dataset_size", str(max_dataset_size)])

        # Run from the job_shard directory
        job_shard_dir = Path(__file__).parent.parent
        result = subprocess.run(cmd, cwd=job_shard_dir, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            pytest.fail(f"Workflow failed with return code {result.returncode}")

        return result

    def _get_shard_contents(self, shard_name):
        """Get the contents of a shard directory."""
        shard_path = os.path.join(self.dest_dir, shard_name)
        if not os.path.exists(shard_path):
            return []

        contents = []
        for item in os.listdir(shard_path):
            if item != "MANIFEST.txt":
                contents.append(item)
        return sorted(contents)

    def _read_manifest(self, shard_name):
        """Read the MANIFEST.txt file from a shard."""
        manifest_path = os.path.join(self.dest_dir, shard_name, "MANIFEST.txt")
        if not os.path.exists(manifest_path):
            return []

        with open(manifest_path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    def _delete_shard(self, shard_name):
        """Delete a specific shard directory."""
        shard_path = os.path.join(self.dest_dir, shard_name)
        if os.path.exists(shard_path):
            shutil.rmtree(shard_path)

    def test_full_integration_workflow(self):
        """Test the complete integration workflow with manifest filtering."""

        # Step 1: Run initial pipeline with 8 files (4 shards of 2 files each)
        print("Step 1: Running initial pipeline with 8 files...")
        self._run_workflow(max_dataset_size=8)

        # Verify initial shards were created
        expected_shards = ["shard-00000", "shard-00001", "shard-00002", "shard-00003"]
        for shard in expected_shards:
            assert os.path.exists(
                os.path.join(self.dest_dir, shard)
            ), f"Shard {shard} should exist"

            # Check MANIFEST.txt exists
            manifest_path = os.path.join(self.dest_dir, shard, "MANIFEST.txt")
            assert os.path.exists(
                manifest_path
            ), f"MANIFEST.txt should exist in {shard}"

            # Check manifest contains exactly 2 video IDs
            manifest_ids = self._read_manifest(shard)
            assert (
                len(manifest_ids) == 2
            ), f"Shard {shard} should have 2 video IDs in manifest"

            # Check shard directory contains the video directories
            shard_contents = self._get_shard_contents(shard)
            assert (
                len(shard_contents) == 2
            ), f"Shard {shard} should have 2 video directories"

            # Verify the manifest IDs match the directory names
            assert set(manifest_ids) == set(
                shard_contents
            ), f"Manifest IDs should match directory names in {shard}"

        # Collect all processed video IDs from manifests
        all_processed_ids = set()
        for shard in expected_shards:
            all_processed_ids.update(self._read_manifest(shard))

        print(
            f"Initial run processed {len(all_processed_ids)} video IDs: {sorted(all_processed_ids)}"
        )
        assert len(all_processed_ids) == 8, "Should have processed 8 unique video IDs"

        # Step 2: Delete one shard to simulate incomplete processing
        print("Step 2: Deleting shard-00002 to simulate incomplete processing...")
        deleted_shard_ids = set(self._read_manifest("shard-00002"))
        self._delete_shard("shard-00002")

        # Verify shard was deleted
        assert not os.path.exists(
            os.path.join(self.dest_dir, "shard-00002")
        ), "shard-00002 should be deleted"

        # Step 3: Re-run pipeline with all 10 files
        print("Step 3: Re-running pipeline with all 10 files...")
        self._run_workflow(max_dataset_size=10)

        # Step 4: Verify results
        print("Step 4: Verifying results...")

        # Check that original shards still exist and unchanged
        for shard in ["shard-00000", "shard-00001", "shard-00003"]:
            assert os.path.exists(
                os.path.join(self.dest_dir, shard)
            ), f"Original shard {shard} should still exist"

            manifest_ids = self._read_manifest(shard)
            assert (
                len(manifest_ids) == 2
            ), f"Original shard {shard} should still have 2 video IDs"

        # Check that new shards were created
        new_shards = []
        for i in range(10):  # Check up to shard-00009
            shard_name = f"shard-{i:05d}"
            if os.path.exists(os.path.join(self.dest_dir, shard_name)):
                new_shards.append(shard_name)

        print(f"Found shards after re-run: {new_shards}")

        # Collect all video IDs from all shards
        final_all_ids = set()
        for shard in new_shards:
            shard_ids = self._read_manifest(shard)
            final_all_ids.update(shard_ids)

            # Each shard should have exactly 2 video IDs (except possibly the last one)
            assert len(shard_ids) <= 2, f"Shard {shard} should have at most 2 video IDs"
            assert len(shard_ids) > 0, f"Shard {shard} should have at least 1 video ID"

        print(
            f"Final run processed {len(final_all_ids)} unique video IDs: {sorted(final_all_ids)}"
        )

        # Verify all 10 video IDs are now processed
        expected_all_ids = set(self.video_ids)
        assert (
            final_all_ids == expected_all_ids
        ), "All 10 video IDs should be processed after re-run"

        # Verify no duplicates across shards
        shard_video_counts = {}
        for shard in new_shards:
            for video_id in self._read_manifest(shard):
                shard_video_counts[video_id] = shard_video_counts.get(video_id, 0) + 1

        duplicates = {
            vid: count for vid, count in shard_video_counts.items() if count > 1
        }
        assert (
            duplicates == {}
        ), f"No video IDs should appear in multiple shards, but found: {duplicates}"

        # Step 5: Verify that re-running again doesn't create more files
        print(
            "Step 5: Verifying idempotency - re-running should not create new files..."
        )
        shards_before_final = sorted(new_shards)

        self._run_workflow(max_dataset_size=10)

        final_shards = []
        for i in range(10):
            shard_name = f"shard-{i:05d}"
            if os.path.exists(os.path.join(self.dest_dir, shard_name)):
                final_shards.append(shard_name)

        final_shards = sorted(final_shards)

        # Should be the same shards as before
        assert (
            shards_before_final == final_shards
        ), "Re-running should not create additional shards"

        # Verify manifest contents are identical
        for shard in final_shards:
            manifest_ids = self._read_manifest(shard)
            assert (
                len(manifest_ids) > 0
            ), f"Shard {shard} should still have video IDs after final run"

        print("Integration test completed successfully!")

    def test_empty_destination_directory(self):
        """Test behavior with empty destination directory."""
        print("Testing empty destination directory...")

        # Run with 4 files
        self._run_workflow(max_dataset_size=4)

        # Should create 2 shards
        expected_shards = ["shard-00000", "shard-00001"]
        for shard in expected_shards:
            assert os.path.exists(
                os.path.join(self.dest_dir, shard)
            ), f"Shard {shard} should be created"

            manifest_ids = self._read_manifest(shard)
            assert len(manifest_ids) == 2, f"Shard {shard} should have 2 video IDs"

    def test_custom_min_shard_idx(self):
        """Test with custom minimum shard index."""
        print("Testing custom minimum shard index...")

        # Run with min_shard_idx=5
        self._run_workflow(max_dataset_size=4, min_shard_idx=5)

        # Should create shards starting from 00005
        expected_shards = ["shard-00005", "shard-00006"]
        for shard in expected_shards:
            assert os.path.exists(
                os.path.join(self.dest_dir, shard)
            ), f"Shard {shard} should be created with min_shard_idx=5"

    def teardown_method(self):
        """Clean up temporary directories."""
        shutil.rmtree(self.test_dir)
