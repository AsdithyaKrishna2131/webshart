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

