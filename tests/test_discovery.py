"""Tests for dataset discovery functionality."""

import json
import tempfile
import pytest
from pathlib import Path
import os

import webshart


def create_test_shard_metadata(shard_num, num_files=100):
    """Create test shard metadata in the expected format."""
    files = {}
    offset = 512  # tar header

    for i in range(num_files):
        filename = f"{shard_num * 1000 + i}.webp"
        size = 1024 * (i % 10 + 1)  # Vary sizes
        files[filename] = {
            "offset": offset,
            "length": size,  # webshart expects 'length' not 'size'
            "sha256": f"deadbeef{i:08x}" * 8,
        }
