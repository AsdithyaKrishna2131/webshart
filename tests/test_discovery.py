"""Tests for dataset discovery functionality."""

import json
import tempfile
import pytest
from pathlib import Path
import os

import webshart


def create_test_shard_metadata(shard_num, num_files=100):
    """Create test shard metadata in the expected format."""
    files = {}
    offset = 512  # tar header

    for i in range(num_files):
        filename = f"{shard_num * 1000 + i}.webp"
        size = 1024 * (i % 10 + 1)  # Vary sizes
        files[filename] = {
            "offset": offset,
            "length": size,  # webshart expects 'length' not 'size'
            "sha256": f"deadbeef{i:08x}" * 8,
        }
        offset += size + 512  # Add tar padding

    return {
        "path": f"data-{shard_num:04d}.tar",
        "filesize": offset,
        "hash": f"hash{shard_num:04d}",
        "hash_lfs": f"lfs_hash{shard_num:04d}",
        "files": files,
    }


def test_discover_local_dataset():
    """Test discovering a local dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock dataset structure
        for i in range(3):
            # Create JSON metadata
            metadata = create_test_shard_metadata(i, num_files=50)
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)

            # Create empty tar file (just for discovery)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        # Discover dataset
        dataset = webshart.discover_dataset(tmpdir)

        assert dataset.name == tmpdir
        assert dataset.num_shards == 3
        assert not dataset.is_remote

        # Check shard info
        shard_info = dataset.get_shard_info(0)
        assert shard_info["name"] == "data-0000"
        assert shard_info["num_files"] == 50

        # List files in first shard
        files = dataset.list_files_in_shard(0)
        assert len(files) == 50
        assert "0.webp" in files

        # Find file location
        shard_idx, local_idx = dataset.find_file_location(75)  # File in second shard
        assert shard_idx == 1
        assert local_idx == 25


def test_discover_local_dataset_recurses_and_preserves_relative_names():
    """Nested local shards should not require flattening or a Hub metadata mirror."""
    with tempfile.TemporaryDirectory() as tmpdir:
        for split in ("train", "validation"):
            shard_dir = Path(tmpdir) / split
            shard_dir.mkdir()
            (shard_dir / "data-0000.tar").touch()
            (shard_dir / "data-0000.json").write_text(
                json.dumps(create_test_shard_metadata(0, num_files=1))
            )

