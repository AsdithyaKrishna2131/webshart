import pytest
from unittest.mock import Mock, MagicMock, patch, create_autospec
import os
import tempfile
import json
from pathlib import Path


class TestBucketDataLoader:
    """Test suite for BucketDataLoader"""

    @pytest.fixture
    def mock_dataset(self):
        """Create a mock discovered dataset"""
        dataset = Mock()
        dataset.inner = Mock()
        dataset.inner.name = "test_dataset"
        dataset.inner.metadata_source = None
        dataset.inner.is_remote = False
        dataset.inner.num_shards = Mock(return_value=2)
        dataset.inner.shards = [
            Mock(tar_path="/path/to/shard1.tar", metadata=Mock()),
            Mock(tar_path="/path/to/shard2.tar", metadata=Mock()),
        ]
        return dataset

    @pytest.fixture
    def mock_file_info(self):
        """Create mock file info objects"""

        def _create_file_info(path, width, height, offset=0, length=1000):
            info = Mock()
            info.path = path
            info.width = width
            info.height = height
            info.aspect = width / height if height > 0 else 1.0
            info.offset = offset
            info.length = length
            return info

        return _create_file_info

    @pytest.fixture
    def temp_dataset(self, tmp_path):
        """Create a temporary local dataset structure"""
        # Create tar files
        (tmp_path / "shard1.tar").write_bytes(b"dummy tar content")
        (tmp_path / "shard2.tar").write_bytes(b"dummy tar content")

        # Create metadata files
        metadata1 = {
            "filesize": 1000,
            "files": {
                "image1.jpg": {
                    "offset": 0,
                    "length": 100,
                    "width": 1920,
                    "height": 1080,
                },
                "image2.jpg": {
                    "offset": 100,
                    "length": 100,
                    "width": 1280,
                    "height": 720,
                },
                "image3.jpg": {
                    "offset": 200,
                    "length": 100,
                    "width": 800,
                    "height": 600,
                },
            },
        }

        metadata2 = {
            "filesize": 1000,
            "files": {
                "image4.jpg": {
                    "offset": 0,
                    "length": 100,
                    "width": 1920,
                    "height": 1080,
                },
                "image5.jpg": {
                    "offset": 100,
                    "length": 100,
                    "width": 3840,
                    "height": 2160,
                },
                "image6.jpg": {
                    "offset": 200,
                    "length": 100,
                    "width": 1024,
                    "height": 1024,
                },
            },
        }

        (tmp_path / "shard1.json").write_text(json.dumps(metadata1))
        (tmp_path / "shard2.json").write_text(json.dumps(metadata2))

        return str(tmp_path)

    @patch("webshart.BucketDataLoader")
    def test_initialization_with_dataset(self, MockBucketDataLoader, mock_dataset):
        """Test initialization with a DiscoveredDataset"""
        # Create a mock instance
        mock_instance = Mock()
        mock_instance.key_type = "aspect"
        mock_sampling_strategy = Mock(name="Sequential")
        mock_instance.sampling_strategy = mock_sampling_strategy
        mock_instance.load_file_data = True
        mock_instance.max_file_size = 50_000_000
        mock_instance.chunk_size_mb = 10
        MockBucketDataLoader.return_value = mock_instance

        # Import and create loader
        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=mock_dataset, key="aspect", sampling_strategy="sequential"
        )

        assert loader.key_type == "aspect"
        assert loader.sampling_strategy.name == mock_sampling_strategy.name
        assert loader.load_file_data is True
        assert loader.max_file_size == 50_000_000
        assert loader.chunk_size_mb == 10

    @patch("webshart.DatasetDiscovery")
    @patch("webshart.BucketDataLoader")
    def test_initialization_with_path(
        self, MockBucketDataLoader, MockDiscovery, temp_dataset
    ):
        """Test initialization with a path string"""
        # Mock the discovery process
        mock_discovered = Mock()
        mock_discovered.name = temp_dataset
        mock_discovered.metadata_source = None
        mock_discovery_instance = Mock()
        mock_discovery_instance.discover_local = Mock(return_value=mock_discovered)
        MockDiscovery.return_value = mock_discovery_instance

        # Create a mock loader instance
        mock_instance = Mock()
        mock_instance.key_type = "geometry-tuple"
        mock_instance.target_pixel_area = 1024 * 1024
        mock_sampling_strategy = Mock(name="RandomWithinBuckets")
        mock_instance.sampling_strategy = mock_sampling_strategy
        MockBucketDataLoader.return_value = mock_instance

        # Import and create loader
        from webshart import BucketDataLoader

        loader = BucketDataLoader(
            dataset_or_path=temp_dataset,
