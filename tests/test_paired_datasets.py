import pytest

import webshart


class FakeDataset:
    def __init__(self, shards):
        self.shards = shards
        self.num_shards = len(shards)

    def list_samples_in_shard(self, shard_index):
        return self.shards[shard_index]


def test_pair_index_joins_by_stem_in_left_order():
    left = FakeDataset([["b.mp3", "a.mp3"], ["nested/c.mp3"]])
    right = FakeDataset([["a.flac"], ["nested/c.wav", "b.wav"]])

    paired = webshart.PairedDataset(left, right)
