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
    entry.aspect = 1.0
    entry.job_id = "shard0000_file000000"
    entry.metadata = {
        "path": "test.jpg",
        "offset": 0,
        "size": 1000,
        "width": 512,
        "height": 512,
        "aspect": 1.0,
    }
    return entry


class TestRangeBasedIteration:
    """Test range-based iteration functionality."""

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_single(self, discovered_dataset):
        """Test setting a single range."""
        loader = TarDataLoader(discovered_dataset)
        loader.set_ranges([(10, 20)])

        # Collect all entries
        entries = list(loader)

        assert len(entries) == 10
        # Check that entries are from the expected range

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_multiple(self, discovered_dataset):
        """Test setting multiple non-overlapping ranges."""
        loader = TarDataLoader(discovered_dataset)
        loader.set_ranges([(10, 20), (50, 60), (100, 110)])

        entries = list(loader)

        assert len(entries) == 30  # 10 + 10 + 10

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "set_ranges"),
        reason="set_ranges not implemented yet",
    )
    def test_set_ranges_validation(self, discovered_dataset):
        """Test range validation."""
        loader = TarDataLoader(discovered_dataset)

        # Invalid range (start >= end)
        with pytest.raises(ValueError, match="Invalid range"):
            loader.set_ranges([(20, 10)])

        # Empty range
        with pytest.raises(ValueError, match="Invalid range"):
            loader.set_ranges([(10, 10)])

    def test_iter_range(self, discovered_dataset):
        """Test iter_range method."""
        loader = TarDataLoader(discovered_dataset)

        # Test normal range
        entries = list(loader.iter_range(start=10, end=20))
        assert len(entries) == 10

        # Test invalid range
        with pytest.raises(ValueError, match="start must be less than end"):
            list(loader.iter_range(start=20, end=10))

    @pytest.mark.skipif(
        not hasattr(TarDataLoader, "iter_range"),
        reason="iter_range not implemented yet",
    )
    def test_range_boundaries_across_shards(self, discovered_dataset):
        """Test ranges that cross shard boundaries."""
        loader = TarDataLoader(discovered_dataset)

        # Range spans from shard 0 (file 90) to shard 1 (file 10)
        entries = list(loader.iter_range(start=90, end=110))

        assert len(entries) == 20


