# test_cache_wait.py
import pytest
import time
import threading
from unittest.mock import Mock, MagicMock, patch
from webshart.cache_wait import CacheWaitContext


class MockDataLoader:
    """Mock dataloader for testing cache wait functionality."""

    def __init__(self, entries, block_pattern=None):
        self.entries = entries
        self.block_pattern = block_pattern or []
        self.index = 0
        self.will_block_calls = 0
        self.prepare_calls = []
        self.current_shard_filename = "shard-0000.tar"
        self._blocking = False
        self._download_start_time = None

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.entries):
            raise StopIteration
        entry = self.entries[self.index]
        self.index += 1
        return entry

    def will_block(self):
        """Returns True based on block pattern."""
        call_idx = self.will_block_calls
        self.will_block_calls += 1
        if call_idx < len(self.block_pattern):
            self._blocking = self.block_pattern[call_idx]
            return self.block_pattern[call_idx]
        return False

    def get_next_shard_info(self):
        return {
            "name": f"shard-{self.index:04d}.tar",
            "index": self.index,
            "size": 1000000,
            "is_cached": False,
        }

    def prepare_next_shard(self):
        self.prepare_calls.append(time.time())
        self._download_start_time = time.time()
        # Simulate that after calling prepare, it takes some time to download
        if self._blocking:
            # Create a thread that will "finish" the download after a delay
            def finish_download():
                time.sleep(0.3)  # Simulate download time
                self._blocking = False

            threading.Thread(target=finish_download, daemon=True).start()

    def get_shard_cache_status(self, shard_name):
        # Simulate download progress
        if self._download_start_time:
            elapsed = time.time() - self._download_start_time
            progress = min(int(elapsed * 3000000), 1000000)  # Fast download for testing
            return {"cur_filesize": progress, "is_cached": progress >= 1000000}
        return {"cur_filesize": 0, "is_cached": False}


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader with test entries."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(10)]
    return MockDataLoader(entries)


def test_cache_wait_context_basic(mock_dataloader):
    """Test basic iteration without blocking."""
    with CacheWaitContext(mock_dataloader, progress_bar=False) as ctx:
        results = list(ctx.iterate())

    assert len(results) == 10
    assert all(hasattr(r, "path") for r in results)


def test_cache_wait_context_with_blocking(mock_dataloader):
    """Test iteration with blocking pattern."""
    # Block on first call only
    mock_dataloader.block_pattern = [True] + [False] * 20

    with CacheWaitContext(mock_dataloader, progress_bar=False) as ctx:
        results = []
        for entry in ctx.iterate():
            results.append(entry)

    assert len(results) == 10
    assert len(mock_dataloader.prepare_calls) >= 1  # Should have prepared at least once


def test_cache_wait_context_progress_bar():
    """Test that progress bars are created and closed properly."""
    entries = [Mock(path=f"file_{i}.jpg") for i in range(5)]
    loader = MockDataLoader(entries, block_pattern=[True] + [False] * 10)

    # Patch tqdm where it's actually used
    with patch("tqdm.tqdm") as mock_tqdm_class:
        mock_pbar = MagicMock()
        mock_cache_pbar = MagicMock()

        # Configure the mock to return different objects
