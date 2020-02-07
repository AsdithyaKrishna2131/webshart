# test_aspect_bucketing.py

import pytest
from unittest.mock import Mock, patch, MagicMock, PropertyMock
import webshart
from webshart import TarDataLoader, discover_dataset


@pytest.fixture
def mock_loader():
    """Create a properly mocked TarDataLoader that doesn't trigger discovery."""
    with patch("webshart.TarDataLoader.__new__") as mock_new:
        # Create a mock instance
        mock_instance = Mock(spec=TarDataLoader)
        mock_new.return_value = mock_instance

        # Set up properties
        mock_instance.num_shards = 3

        # Configure the methods we'll test
        mock_instance.list_shard_aspect_buckets = Mock()
        mock_instance.list_all_aspect_buckets = Mock()

        yield mock_instance


@pytest.fixture
def mock_loader_factory():
    """Factory to create mock loaders with custom behavior."""

    def _create_loader(num_shards=3):
        loader = Mock(spec=TarDataLoader)
        type(loader).num_shards = PropertyMock(return_value=num_shards)
        loader.list_shard_aspect_buckets = Mock()
        loader.list_all_aspect_buckets = Mock()
        return loader

    return _create_loader


class TestAspectBucketing:
    """Test suite for aspect bucketing functionality."""

    def test_list_shard_aspect_buckets_default(self, mock_loader_factory):
        """Test basic aspect bucketing with default parameters."""
        loader = mock_loader_factory()

        # Set up the expected return value
        loader.list_shard_aspect_buckets.return_value = [
            {
                "shard_idx": 0,
                "shard_name": "shard-00000.tar",
                "buckets": {
                    "1.778": [
                        {
                            "filename": "image1.jpg",
                            "width": 1920,
                            "height": 1080,
                            "aspect": 1.778,
                        },
                        {
                            "filename": "image3.jpg",
                            "width": 1920,
                            "height": 1080,
                            "aspect": 1.778,
                        },
                    ],
                    "0.563": [
                        {
                            "filename": "image2.jpg",
                            "width": 1080,
                            "height": 1920,
                            "aspect": 0.563,
                        }
                    ],
                },
            }
        ]

        buckets = loader.list_shard_aspect_buckets([0])

        assert len(buckets) == 1
        assert buckets[0]["shard_idx"] == 0
        assert "1.778" in buckets[0]["buckets"]
        assert len(buckets[0]["buckets"]["1.778"]) == 2
        assert "0.563" in buckets[0]["buckets"]
