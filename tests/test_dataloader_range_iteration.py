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
                        "length": 1000,
                        "width": 512,
                        "height": 512,
                        "aspect": 1.0,
                    }
                )

            metadata = {"version": 2, "filesize": 100000, "files": files}

            json_path = Path(tmpdir) / f"shard-{shard_idx:04d}.json"
            json_path.write_text(json.dumps(metadata))

        yield str(tmpdir)


@pytest.fixture
def discovered_dataset(test_dataset_dir):
    """Create a discovered dataset from test directory."""
    discovery = DatasetDiscovery()
    return discovery.discover_local(test_dataset_dir)


@pytest.fixture
def mock_tar_entry():
    """Create a mock TarFileEntry for testing."""
    from types import SimpleNamespace

    entry = SimpleNamespace()
    entry.path = "test.jpg"
    entry.offset = 0
    entry.size = 1000
    entry.data = b"test data"
    entry.width = 512
    entry.height = 512
