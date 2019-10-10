"""Rolling conversion of loose or legacy tar datasets into webshart shards."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import Any, BinaryIO, Iterator, Optional, Sequence, Union
import json
import os
import shutil
import tarfile
import urllib.request

from tqdm import tqdm
from webshart._webshart import MetadataExtractor


CaptionValue = Union[str, list[str]]

DEFAULT_PAYLOAD_EXTENSIONS = (
    ".avif",
    ".bmp",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".jxl",
    ".m4a",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".tif",
    ".tiff",
    ".wav",
    ".webm",
    ".webp",
)
CAPTION_KEYS = (
    "caption",
    "captions",
    "text",
    "txt",
    "description",
    "descriptions",
    "prompt",
    "alt_text",
)
STATE_FILENAME = ".webshart-optimize-state.json"
STATE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SourceFile:
    path: str
    size: int
    local_path: Optional[Path] = None


@dataclass(frozen=True)
class LooseSample:
    path: str
    size: int
    payload: SourceFile
    sidecar: Optional[SourceFile]


@dataclass
class OptimizationState:
    schema_version: int
    status: str
    source: dict[str, Any]
    manifest_sha256: str
    output_prefix: str
    max_shard_size_bytes: int
    payload_extensions: list[str]
    total_samples: int
    input_layout: str = "loose"
    next_sample_index: int = 0
    next_shard_index: int = 0
    next_source_archive_index: int = 0
    next_source_member_offset: int = 0
    captioned_samples: int = 0
    uncaptioned_samples: int = 0
    bytes_sharded: int = 0


def _require_hub():
    try:
        from huggingface_hub import (
            CommitOperationAdd,
            HfApi,
            get_hf_file_metadata,
            hf_hub_download,
            hf_hub_url,
        )
    except ImportError as exc:
        raise ImportError(
            "Hub optimization requires huggingface-hub; install webshart[hub]"
        ) from exc
    return (
        HfApi,
        CommitOperationAdd,
        hf_hub_download,
        hf_hub_url,
        get_hf_file_metadata,
    )


def _normalize_prefix(value: str) -> str:
    value = value.strip("/")
    if not value:
        return ""
    path = PurePosixPath(value)
    if any(part in {"", ".", ".."} for part in path.parts):
