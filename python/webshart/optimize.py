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
        raise ValueError(f"invalid repository path prefix: {value!r}")
    return str(path)


def _repo_path(prefix: str, filename: str) -> str:
    return str(PurePosixPath(prefix, filename)) if prefix else filename


def _normalize_extensions(extensions: Sequence[str]) -> tuple[str, ...]:
    normalized = []
    for extension in extensions:
        extension = extension.strip().lower()
        if not extension:
            continue
        normalized.append(extension if extension.startswith(".") else f".{extension}")
    if not normalized:
        raise ValueError("at least one payload extension is required")
    return tuple(sorted(set(normalized)))


def _relative_source_path(path: str, subfolder: str) -> Optional[str]:
    source_path = PurePosixPath(path)
    if source_path.is_absolute() or ".." in source_path.parts:
        raise ValueError(f"unsafe source path: {path!r}")
    if not subfolder:
        return str(source_path)
    prefix = PurePosixPath(subfolder)
    try:
        return str(source_path.relative_to(prefix))
    except ValueError:
        return None


def _list_local_files(source: Path, subfolder: str) -> list[SourceFile]:
    root = source / Path(subfolder) if subfolder else source
    if not root.is_dir():
        raise ValueError(f"local source folder does not exist: {root}")
    files = []
    for path in root.rglob("*"):
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            files.append(SourceFile(relative, path.stat().st_size, path))
    return files


def _list_hub_files(
    repo_id: str,
    subfolder: str,
    revision: str,
    token: Optional[str],
) -> tuple[list[SourceFile], str]:
    HfApi, _, _, _, _ = _require_hub()
    api = HfApi(token=token)
    info = api.dataset_info(
        repo_id,
        revision=revision,
        token=token,
        files_metadata=False,
    )
    files = []
    for entry in info.siblings or ():
        relative = _relative_source_path(entry.rfilename, subfolder)
        if relative is not None:
            files.append(SourceFile(relative, -1))
    return files, info.sha


def _build_samples(
    files: Sequence[SourceFile], payload_extensions: Sequence[str]
) -> list[LooseSample]:
    sidecars: dict[str, SourceFile] = {}
    for file in files:
        path = PurePosixPath(file.path)
        extension = path.suffix.lower()
        if extension not in {".txt", ".json"}:
            continue
        stem = str(path.with_suffix(""))
        existing = sidecars.get(stem)
        if existing is None or extension == ".txt":
            sidecars[stem] = file
    payloads = [
        file
        for file in files
        if PurePosixPath(file.path).suffix.lower() in payload_extensions
    ]
    samples = []
    for payload in sorted(payloads, key=lambda file: file.path):
        stem = str(PurePosixPath(payload.path).with_suffix(""))
        sidecar = sidecars.get(stem)
        samples.append(
            LooseSample(
                path=payload.path,
                size=payload.size,
