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
