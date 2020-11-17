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
