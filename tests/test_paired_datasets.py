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

    assert [pair.key for pair in paired.list_pairs()] == ["b", "a", "nested/c"]
    assert paired.get_pair(0).left.sample_index == 0
    assert paired.get_pair(0).right == webshart.SampleLocation(1, 1, "b.wav")


def test_pair_index_reports_mismatches_in_strict_mode():
    paired = webshart.PairedDataset(
        FakeDataset([["shared.mp3", "left-only.mp3"]]),
        FakeDataset([["shared.mp3", "right-only.mp3"]]),
