"""
Tests for webshart dataset statistics methods
"""

import pytest
import tempfile
import json
import os
from unittest.mock import patch, MagicMock
from webshart import discover_dataset, DatasetDiscovery


class TestDatasetStats:
    """Test dataset statistics and info methods."""

    @pytest.fixture
    def mock_dataset_dir(self, tmp_path):
        """Create a mock dataset directory with tar and json files."""
        # Create mock tar files
        for i in range(3):
            tar_path = tmp_path / f"shard-{i:04d}.tar"
            tar_path.write_bytes(b"mock tar content")

            # Create corresponding JSON metadata
            metadata = {
                "path": f"shard-{i:04d}.tar",
                "filesize": 1000000 * (i + 1),  # 1MB, 2MB, 3MB
                "files": {
                    f"image_{j:03d}.webp": {
                        "offset": j * 1000,
                        "length": 950,
                        "width": 1920,
                        "height": 1080,
                        "aspect": 1.7777778,
                    }
                    for j in range(10 * (i + 1))  # 10, 20, 30 files
                },
                "includes_image_geometry": True,
            }
            json_path = tmp_path / f"shard-{i:04d}.json"
            json_path.write_text(json.dumps(metadata))

        return tmp_path

    @pytest.fixture
    def local_dataset(self, mock_dataset_dir):
        """Create a discovered dataset from mock directory."""
        return discover_dataset(str(mock_dataset_dir))

    def test_get_shard_file_count(self, local_dataset):
        """Test getting file count for specific shard."""
        # Test valid indices
        assert local_dataset.get_shard_file_count(0) == 10
        assert local_dataset.get_shard_file_count(1) == 20
        assert local_dataset.get_shard_file_count(2) == 30

        # Test out of range
        with pytest.raises(IndexError) as exc_info:
            local_dataset.get_shard_file_count(3)
        assert "out of range" in str(exc_info.value)

    def test_get_total_files(self, local_dataset):
        """Test getting total file count."""
        # This loads all metadata
        total = local_dataset.total_files
        assert total == 60  # 10 + 20 + 30

    def test_stats_property(self, local_dataset):
        """Test the stats property getter."""
        stats = local_dataset.get_stats()

        assert stats["total_shards"] == 3
        assert "shard_details" in stats
        assert len(stats["shard_details"]) == 3

        # First access won't have loaded metadata
        assert stats["shard_details"][0]["metadata_loaded"] == False

        # Load a shard
        local_dataset.get_shard_file_count(0)

        # Check again
        stats = local_dataset.get_stats()
        assert stats["shard_details"][0]["metadata_loaded"] == True
        assert stats["shard_details"][0]["num_files"] == 10

    def test_detailed_stats_property(self, local_dataset):
        """Test the detailed_stats property getter."""
        stats = local_dataset.get_detailed_stats()

        # Should load all metadata
        assert stats["total_shards"] == 3
        assert stats["total_files"] == 60
        assert stats["total_size"] == 6000000  # 1MB + 2MB + 3MB
        assert stats["total_size_gb"] == pytest.approx(0.00559, rel=0.01)
        assert stats["average_files_per_shard"] == 20.0
        assert stats["min_files_in_shard"] == 10
        assert stats["max_files_in_shard"] == 30
        assert stats["from_cache"] == False

        # Check shard details
        assert len(stats["shard_details"]) == 3
