import pytest
import webshart
import tempfile
import json
import os
import tarfile
from pathlib import Path
import io


@pytest.fixture
def mock_dataset_dir():
    """Create a temporary directory with proper webshart-compatible dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create mock tar files with proper structure
        for i in range(3):
            shard_name = f"shard_{i:04d}"
            tar_path = Path(tmpdir) / f"{shard_name}.tar"

            # Create a proper tar file
            with tarfile.open(tar_path, "w") as tar:
                # Add some dummy files
                for j in range(10):
                    filename = f"{shard_name}_{j:06d}.jpg"
                    # Create dummy image data
                    data = f"FAKE_JPEG_DATA_SHARD{i}_FILE{j}".encode() * 100

                    # Create tarinfo
                    info = tarfile.TarInfo(name=filename)
                    info.size = len(data)

                    # Add to tar
                    tar.addfile(info, io.BytesIO(data))

            # Create metadata JSON that matches the tar structure
            metadata = {"filesize": os.path.getsize(tar_path), "files": {}}

            # Read the tar to get actual offsets
            with tarfile.open(tar_path, "r") as tar:
                offset = 0
                for member in tar:
                    if member.isfile():
                        # In tar files, the header is 512 bytes, followed by data rounded up to 512 bytes
                        header_size = 512
                        data_blocks = (member.size + 511) // 512
                        data_size = data_blocks * 512

                        metadata["files"][member.name] = {
                            "offset": offset + header_size,  # Offset to actual data
                            "length": member.size,  # Actual file size
                        }
                        offset += header_size + data_size

            json_path = Path(tmpdir) / f"{shard_name}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)

        yield tmpdir


@pytest.fixture
def discovered_dataset(mock_dataset_dir):
    """Create a discovered dataset from mock directory."""
    return webshart.discover_dataset(mock_dataset_dir)


class TestTarDataLoaderStateDict:
    """Test state dict functionality for resumable pipelines."""

    def test_state_dict_basic(self, discovered_dataset):
        """Test basic state dict saving."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read a few files
        for _ in range(3):
            next(loader)

        # Get state
        state = loader.state_dict()

        # Check state contents
        assert isinstance(state, dict)
        assert state["current_shard"] == 0
        assert state["current_file_index"] == 3
        assert state["buffer_size"] == 5
        assert state["version"] == 4
        assert "source" in state
        assert "num_shards" in state

    def test_load_state_dict(self, discovered_dataset):
        """Test loading state dict."""
        loader1 = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read some files and switch shards
        for _ in range(15):  # Read past first shard
            next(loader1)

        # Save state
        state = loader1.state_dict()

        # Create new loader and load state
