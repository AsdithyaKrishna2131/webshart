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
