import pytest

import webshart


class FakeDataset:
    def __init__(self, shards):
        self.shards = shards
        self.num_shards = len(shards)
