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
