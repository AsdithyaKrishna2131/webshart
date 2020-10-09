"""Tests for metadata handling in webshart."""

import json
import sys
import tempfile
import tarfile
import types
import pytest
from pathlib import Path
from io import BytesIO

import webshart


def create_hashmap_metadata():
    """Create metadata in HashMap format."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "hash": "abcdef123456",
        "hash_lfs": "fedcba654321",
        "files": {
            "image_001.webp": {
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            "image_002.webp": {
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            "image_003.webp": {
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        },
    }


def create_vec_metadata():
    """Create metadata in Vec/Array format (alternative format)."""
    return {
        "path": "test-shard.tar",
        "filesize": 1048576,
        "files": [
            {
                "path": "image_001.webp",
                "offset": 512,
                "length": 102400,
                "sha256": "deadbeef" * 8,
            },
            {
                "path": "image_002.webp",
