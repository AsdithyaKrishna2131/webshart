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
