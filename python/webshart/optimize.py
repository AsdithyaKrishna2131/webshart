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
                payload=payload,
                sidecar=sidecar,
            )
        )
    return samples


def _manifest_sha256(samples: Sequence[LooseSample]) -> str:
    digest = sha256()
    for sample in samples:
        digest.update(sample.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(sample.size).encode("ascii"))
        digest.update(b"\0")
        if sample.sidecar is not None:
            digest.update(sample.sidecar.path.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(sample.sidecar.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _file_manifest_sha256(files: Sequence[SourceFile]) -> str:
    digest = sha256()
    for file in files:
        digest.update(file.path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(file.size).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


@contextmanager
def _open_source_file(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
    offset: int = 0,
) -> Iterator[BinaryIO]:
    if file.local_path is not None:
        with file.local_path.open("rb") as handle:
            if offset:
                handle.seek(offset)
            yield handle
        return

    if source_repo is None:
        raise ValueError("remote source repository is missing")
    _, _, _, hf_hub_url, _ = _require_hub()
    remote_path = _repo_path(source_subfolder, file.path)
    url = hf_hub_url(
        source_repo,
        remote_path,
        repo_type="dataset",
        revision=source_revision,
    )
    headers = {"Accept-Encoding": "identity", "User-Agent": "webshart/optimize-dataset"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request) as response:
        if offset and getattr(response, "status", None) != 206:
            raise ValueError(
                f"source server ignored the resume range for {file.path!r}"
            )
        yield response


def _source_file_size(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> int:
    if file.size >= 0:
        return file.size
    if source_repo is None:
        raise ValueError(f"source size is unavailable: {file.path}")
    _, _, _, hf_hub_url, get_hf_file_metadata = _require_hub()
    remote_path = _repo_path(source_subfolder, file.path)
    url = hf_hub_url(
        source_repo,
        remote_path,
        repo_type="dataset",
        revision=source_revision,
    )
    metadata = get_hf_file_metadata(url, token=token)
    if metadata.size is None:
        raise ValueError(f"Hub did not report a size for {remote_path}")
    return int(metadata.size)


def _read_small_file(
    file: SourceFile,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
    max_bytes: int = 4 * 1024 * 1024,
) -> bytes:
    if file.size > max_bytes:
        raise ValueError(f"caption sidecar exceeds {max_bytes} bytes: {file.path}")
    with _open_source_file(
        file,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    ) as handle:
        data = handle.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"caption sidecar exceeds {max_bytes} bytes: {file.path}")
    return data


def _normalize_caption(value: Any) -> Optional[CaptionValue]:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, list):
        captions = []
        seen = set()
        for item in value:
            if isinstance(item, str):
                item = item.strip()
                if item and item not in seen:
                    seen.add(item)
                    captions.append(item)
        if len(captions) == 1:
            return captions[0]
        return captions or None
    return None


def _metadata_from_sidecar(
    sidecar: Optional[SourceFile],
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> tuple[Optional[CaptionValue], Optional[dict[str, Any]]]:
    if sidecar is None:
        return None, None
    data = _read_small_file(
        sidecar,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    )
    if PurePosixPath(sidecar.path).suffix.lower() == ".txt":
        return _normalize_caption(data.decode("utf-8")), None

    value = json.loads(data)
    if not isinstance(value, dict):
        return None, None
    captions = []
    for key in CAPTION_KEYS:
        caption = _normalize_caption(value.get(key))
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, list):
            captions.extend(caption)
    return _normalize_caption(captions), value


def _tar_member_size(payload_size: int) -> int:
    return 512 + ((payload_size + 511) // 512) * 512


def _add_to_tar(
    archive: tarfile.TarFile,
    sample: LooseSample,
    payload_size: int,
    *,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
) -> None:
    info = tarfile.TarInfo(sample.path)
    info.size = payload_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    with _open_source_file(
        sample.payload,
        source_repo=source_repo,
        source_subfolder=source_subfolder,
        source_revision=source_revision,
        token=token,
    ) as handle:
        archive.addfile(info, handle)


def _normalized_tar_member_path(name: str) -> Optional[str]:
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"unsafe path in source tar: {name!r}")
    parts = [part for part in path.parts if part not in {"", "."}]
    return str(PurePosixPath(*parts)) if parts else None


def _filename_caption(path: str) -> Optional[str]:
    caption = PurePosixPath(path).stem.replace("_", " ").strip()
    return caption or None


def _write_legacy_tar_shard(
    tar_path: Path,
    archives: Sequence[SourceFile],
    state: OptimizationState,
    *,
    payload_extensions: Sequence[str],
    max_shard_size_bytes: int,
    source_repo: Optional[str],
    source_subfolder: str,
    source_revision: str,
    token: Optional[str],
    progress: Any,
) -> tuple[int, dict[str, CaptionValue]]:
    """Stream legacy tar members into one deterministic output shard."""
    captions: dict[str, CaptionValue] = {}
    shard_size = 1024
    shard_samples = 0
    shard_full = False

    with tarfile.open(tar_path, mode="w", format=tarfile.PAX_FORMAT) as output:
        while state.next_source_archive_index < len(archives) and not shard_full:
            source_archive = archives[state.next_source_archive_index]
            source_archive_size = _source_file_size(
                source_archive,
                source_repo=source_repo,
                source_subfolder=source_subfolder,
                source_revision=source_revision,
                token=token,
            )
            base_offset = state.next_source_member_offset
            with _open_source_file(
                source_archive,
                source_repo=source_repo,
                source_subfolder=source_subfolder,
                source_revision=source_revision,
                token=token,
                offset=base_offset,
            ) as source_handle:
                with tarfile.open(fileobj=source_handle, mode="r|") as source_tar:
                    for member in source_tar:
                        member_offset = base_offset + member.offset
                        next_offset = (
                            base_offset
                            + member.offset_data
                            + ((member.size + 511) // 512) * 512
                        )
                        if next_offset > source_archive_size:
                            progress.write(
                                "Skipping truncated tail member "
                                f"{member.name!r} in {source_archive.path!r}"
                            )
                            break
                        member_path = _normalized_tar_member_path(member.name)
                        if (
                            not member.isfile()
                            or member_path is None
                            or PurePosixPath(member_path).suffix.lower()
                            not in payload_extensions
                        ):
                            state.next_source_member_offset = next_offset
                            continue

                        member_size = _tar_member_size(member.size)
                        if (
                            shard_samples
                            and shard_size + member_size > max_shard_size_bytes
                        ):
                            state.next_source_member_offset = member_offset
                            shard_full = True
                            break

                        archive_prefix = str(
                            PurePosixPath(source_archive.path).with_suffix("")
                        )
                        output_path = str(PurePosixPath(archive_prefix, member_path))
                        info = tarfile.TarInfo(output_path)
                        info.size = member.size
                        info.mode = 0o644
                        info.mtime = 0
                        info.uid = 0
                        info.gid = 0
                        info.uname = ""
                        info.gname = ""
                        payload = source_tar.extractfile(member)
                        if payload is None:
                            raise ValueError(
                                f"unable to read {member.name!r} from {source_archive.path!r}"
                            )
                        output.addfile(info, payload)

                        caption = _filename_caption(member_path)
                        if caption is not None:
                            captions[output_path] = caption
                            state.captioned_samples += 1
                        else:
                            state.uncaptioned_samples += 1
                        state.next_sample_index += 1
                        state.bytes_sharded += member.size
                        state.next_source_member_offset = next_offset
                        shard_size += member_size
                        shard_samples += 1
                        progress.update(1)

            if not shard_full:
                state.next_source_archive_index += 1
                state.next_source_member_offset = 0

    return shard_samples, captions


def _apply_sidecar_metadata(
    metadata_path: Path,
    captions: dict[str, CaptionValue],
    json_metadata: dict[str, dict[str, Any]],
) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    files = metadata.get("files")
    if not isinstance(files, dict):
        raise ValueError(f"invalid webshart metadata generated at {metadata_path}")
    for path, caption in captions.items():
        entry = files.get(path)
        if isinstance(entry, dict):
            entry.pop("caption", None)
            entry["captions"] = caption
    for path, value in json_metadata.items():
        entry = files.get(path)
        if isinstance(entry, dict):
            entry["json_metadata"] = value
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )


def _write_state(path: Path, state: OptimizationState) -> None:
    path.write_text(
        json.dumps(asdict(state), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def _load_local_state(path: Path) -> Optional[OptimizationState]:
    if not path.is_file():
        return None
    return OptimizationState(**json.loads(path.read_text(encoding="utf-8")))


def _load_hub_state(
    api: Any,
    repo_id: str,
    state_repo_path: str,
    revision: str,
    token: Optional[str],
) -> Optional[OptimizationState]:
    _, _, hf_hub_download, _, _ = _require_hub()
    if not _hub_file_exists(
        api,
        repo_id,
        state_repo_path,
        revision=revision,
        token=token,
    ):
        return None
    state_path = hf_hub_download(
        repo_id,
        state_repo_path,
        repo_type="dataset",
        revision=revision,
        token=token,
    )
    return _load_local_state(Path(state_path))


def _hub_file_exists(
    api: Any,
    repo_id: str,
    path: str,
    *,
    revision: str,
    token: Optional[str],
) -> bool:
    try:
        return api.file_exists(
            repo_id,
            path,
            repo_type="dataset",
            revision=revision,
            token=token,
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        if getattr(response, "status_code", None) == 404:
            return False
        raise


def _validate_resume_state(
    state: OptimizationState,
    expected: OptimizationState,
) -> None:
    immutable_fields = (
        "schema_version",
