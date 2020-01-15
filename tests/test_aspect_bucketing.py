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
