"""Tests for metadata handling in webshart."""

import json
import sys
import tempfile
import tarfile
import types
import pytest
from pathlib import Path
from io import BytesIO

import webshart


def create_hashmap_metadata():
    """Create metadata in HashMap format."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "hash": "abcdef123456",
        "hash_lfs": "fedcba654321",
        "files": {
            "image_001.webp": {
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            "image_002.webp": {
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            "image_003.webp": {
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        },
    }


def create_vec_metadata():
    """Create metadata in Vec/Array format (alternative format)."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "files": [
            {
                "path": "image_001.webp",
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            {
                "path": "image_002.webp",
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            {
                "path": "image_003.webp",
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        ],
    }


def test_metadata_hashmap_format():
    """Test loading metadata in HashMap format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with HashMap metadata
        metadata = create_hashmap_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576

        # Check file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 3
        assert "image_001.webp" in files
        assert "image_002.webp" in files
        assert "image_003.webp" in files


def test_metadata_vec_format():
    """Test loading metadata in Vec/Array format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with Vec metadata
        metadata = create_vec_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576
