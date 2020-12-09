import pytest
import tempfile
import shutil
import json
import os
from pathlib import Path
import time
from unittest.mock import Mock, patch, MagicMock

# These would be the actual Python imports from webshart
# from webshart import DatasetDiscovery, DiscoveredDataset


class TestMetadataCachingPythonAPI:
    """Test suite for metadata caching through Python API"""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def temp_dataset_dir(self):
        """Create a temporary directory with mock dataset files"""
        temp_dir = tempfile.mkdtemp()

        # Create mock shard files
        for i in range(5):
            # Create tar file
            tar_path = os.path.join(temp_dir, f"shard-{i:04d}.tar")
            with open(tar_path, "wb") as f:
                f.write(b"mock tar content")

            # Create metadata json
            metadata = {
                "filesize": 1024 * (i + 1),
                "files": {
                    f"image_{i}_001.webp": {"offset": 0, "length": 512},
                    f"image_{i}_002.webp": {"offset": 512, "length": 512},
                },
            }
            json_path = os.path.join(temp_dir, f"shard-{i:04d}.json")
            with open(json_path, "w") as f:
                json.dump(metadata, f)

        yield temp_dir
        shutil.rmtree(temp_dir)

    def test_enable_metadata_cache_basic(self, temp_cache_dir, temp_dataset_dir):
        """Test basic cache enabling functionality"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Should work without cache
        info = dataset.get_shard_info(0)
        assert info["name"] == "shard-0000"

        # Enable cache
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=2)

        # Cache should be enabled
        stats = dataset.get_cache_stats()
        assert stats["cache_enabled"] is True
        assert "cache_location" in stats
        assert stats["cached_shards"] >= 0  # May have pre-loaded some

    def test_cache_stats_accuracy(self, temp_cache_dir, temp_dataset_dir):
        """Test that cache statistics are accurate"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Enable cache with no pre-loading
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=0)

        # Initially no cached shards
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 0
        assert stats["cache_size_bytes"] == 0

        # Access some shards
        dataset.get_shard_info(0)
        dataset.get_shard_info(1)

        # Check stats updated
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 2
        assert stats["cache_size_bytes"] > 0
        assert stats["cache_size_mb"] > 0

    def test_preload_on_enable(self, temp_cache_dir, temp_dataset_dir):
        """Test that init_shard_count pre-loads metadata"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()
        dataset = discovery.discover_local(temp_dataset_dir)

        # Enable cache with pre-loading
        dataset.enable_metadata_cache(temp_cache_dir, init_shard_count=3)

        # Should have cached 3 shards
        stats = dataset.get_cache_stats()
        assert stats["cached_shards"] == 3

    def test_cache_persistence(self, temp_cache_dir, temp_dataset_dir):
        """Test that cache persists across dataset instances"""
        from webshart import DatasetDiscovery

        discovery = DatasetDiscovery()

        # First instance - populate cache
        dataset1 = discovery.discover_local(temp_dataset_dir)
