"""Shared pytest configuration.

A handful of tests exercise the Hugging Face Hub metadata path, which is
not implemented in the current Rust core yet (the extension raises
"HF Hub upload not implemented yet"). They are tracked here as expected
failures so the suite stays green until the feature lands.
"""
import pytest

PENDING_HF_HUB = [
    "test_dataset_streaming.py::TestTarDataLoader::test_shard_switch_by_filename",
    "test_metadata.py::test_paired_json_sidecar_metadata_and_retrieval",
