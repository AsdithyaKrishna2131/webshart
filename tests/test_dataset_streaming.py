# test_tar_dataloader.py
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


@pytest.fixture
def mixed_size_dataset_dir():
    """Create one shard with oversized files interleaved with eligible files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "mixed-0000.tar"
        contents = {
            "0-oversized.jpg": b"x" * 101,
            "1-small.jpg": b"small-zero",
            "2-oversized.jpg": b"y" * 150,
            "3-small.jpg": b"z" * 100,
        }

        with tarfile.open(tar_path, "w") as tar:
            for filename, data in contents.items():
                info = tarfile.TarInfo(name=filename)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

        with tarfile.open(tar_path, "r") as tar:
            files = {
                member.name: {
                    "offset": member.offset_data,
                    "length": member.size,
                    "width": 64,
                    "height": 64,
                    "aspect": 1.0,
                }
                for member in tar
                if member.isfile()
            }

        (Path(tmpdir) / "mixed-0000.json").write_text(
            json.dumps({"filesize": tar_path.stat().st_size, "files": files}),
            encoding="utf-8",
        )
        yield tmpdir


class TestTarDataLoader:
    """Test the TarDataLoader functionality."""

    def test_basic_iteration(self, discovered_dataset):
        """Test basic iteration through files."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        files_read = []
        count = 0
        for entry in loader:
            files_read.append(entry.path)
            count += 1
            if count >= 10:  # Read first 10 files
                break

        assert len(files_read) == 10
        # Files should start with shard_0000
        assert all(f.startswith("shard_0000") for f in files_read)

    def test_skip_within_shard(self, discovered_dataset):
        """Test skipping files within a shard."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Skip first 5 files
        loader.skip(5)

        # Next file should be the 6th file (index 5)
        entry = next(loader)
        assert "000005" in entry.path  # Should be file 5 (0-indexed)

    def test_skip_to_end_of_shard(self, discovered_dataset):
        """Test skipping to near end of shard."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Skip to file 8 (9th file)
        loader.skip(8)

        # Should be able to read files 8 and 9
        entries = []
        for entry in loader:
            entries.append(entry)
            if len(entries) >= 2:
                break

        assert len(entries) == 2
        assert "000008" in entries[0].path
        assert "000009" in entries[1].path

    def test_shard_switch_by_index(self, discovered_dataset):
        """Test switching shards by index."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 1
        loader.shard(shard_idx=1)

        # Next file should be from shard 1
        entry = next(loader)
        assert "shard_0001" in entry.path
        assert "000000" in entry.path  # First file in shard

    def test_shard_switch_by_filename(self, discovered_dataset):
        """Test switching shards by filename."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 2 by filename
        loader.shard(filename="shard_0002")

        # Next file should be from shard 2
        entry = next(loader)
        assert "shard_0002" in entry.path

    def test_shard_switch_with_cursor(self, discovered_dataset):
        """Test switching shards with cursor position."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Switch to shard 1, starting at file 3
        loader.shard(shard_idx=1, cursor_idx=3)

        # Next file should be file 3 from shard 1
        entry = next(loader)
        assert "shard_0001" in entry.path
        assert "000003" in entry.path

    def test_reset(self, discovered_dataset):
        """Test resetting the loader."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=5)

        # Read a few files
        for _ in range(5):
            next(loader)

        # Reset
        loader.reset()

        # Should start from beginning again
        entry = next(loader)
        assert "shard_0000" in entry.path
        assert "000000" in entry.path

    def test_buffer_size_property(self, discovered_dataset):
        """Test buffer size property."""
        loader = webshart.TarDataLoader(discovered_dataset, buffer_size=10)

        assert loader.buffer_size == 10

        # Change buffer size
        loader.buffer_size = 20
        assert loader.buffer_size == 20

    def test_num_shards_property(self, discovered_dataset):
        """Test num_shards property."""
