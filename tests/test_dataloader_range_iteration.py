# test_range_iteration.py

import pytest
from webshart import TarDataLoader, DiscoveredDataset, DatasetDiscovery
import tempfile
import json
import os
from pathlib import Path


@pytest.fixture
def test_dataset_dir():
    """Create a temporary dataset directory with tar files and metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock tar files and metadata
        for shard_idx in range(3):
            # Create empty tar file
            tar_path = Path(tmpdir) / f"shard-{shard_idx:04d}.tar"
            tar_path.write_bytes(b"mock tar data")

            # Create metadata JSON
            files = []
            for file_idx in range(100):
                files.append(
                    {
                        "path": f"shard{shard_idx}_file{file_idx}.jpg",
                        "offset": file_idx * 1000,
