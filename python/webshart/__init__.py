"""Fast and memory-efficient webdataset shard reader with synchronous and batch support."""

from pathlib import Path
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import (
    Optional,
    Union,
    List,
    Tuple,
    Any,
    Dict,
    Mapping,
    MutableMapping,
    Callable,
    Iterator,
)
import argparse
import os
import sys
import json
from .cache_wait import CacheWaitContext, iter_with_cache_wait, next_with_cache_wait


from webshart._webshart import (
    __version__,
    DatasetDiscovery,
    DiscoveredDataset,
    BatchOperations,
    MetadataExtractor,
    TarDataLoader,
    BucketDataLoader,
)
from .optimize import DEFAULT_PAYLOAD_EXTENSIONS, optimize_dataset

__all__ = [
    "__version__",
    "DatasetDiscovery",
    "DiscoveredDataset",
    "MetadataExtractor",
    "TarDataLoader",
    "BucketDataLoader",
    "discover_dataset",
    "discover_paired_dataset",
    "PairedDataset",
    "PairedTarDataLoader",
    "SampleLocation",
    "SamplePair",
    "LoadedSamplePair",
    "BatchOperations",
    "discover_datasets_batch",
    "read_files_batch",
    "CacheWaitContext",
    "iter_with_cache_wait",
    "next_with_cache_wait",
    "apply_captions_to_metadata",
    "write_captions_to_metadata",
    "upload_caption_metadata",
    "optimize_dataset",
]


CaptionValue = Union[str, List[str]]
OptionalCaptionValue = Optional[CaptionValue]


@dataclass(frozen=True)
class SampleLocation:
    """Stable location of one logical sample in a discovered dataset."""

    shard_index: int
    sample_index: int
    filename: str


@dataclass(frozen=True)
class SamplePair:
    """Two logical samples joined by a shared key."""

    key: str
    left: SampleLocation
    right: SampleLocation


@dataclass(frozen=True)
class LoadedSamplePair:
    """Loaded values for a :class:`SamplePair`."""

    key: str
    left: Any
    right: Any


def _default_pair_key(filename: str) -> str:
    return str(PurePosixPath(filename).with_suffix(""))


class PairedDataset:
    """Opt-in key join over two independently discovered datasets.

    The underlying datasets and their normal loader behavior are unchanged.
    Pair locations are indexed lazily the first time they are requested.
    """

    def __init__(
        self,
        left: DiscoveredDataset,
        right: DiscoveredDataset,
        *,
        strict: bool = True,
        pair_key: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.left = left
        self.right = right
        self.strict = strict
        self.pair_key = pair_key or _default_pair_key
        self._pairs: Optional[List[SamplePair]] = None
        self._unmatched_left: Optional[List[str]] = None
        self._unmatched_right: Optional[List[str]] = None

    def _sample_locations(
        self, dataset: DiscoveredDataset, side: str
    ) -> Dict[str, SampleLocation]:
        locations: Dict[str, SampleLocation] = {}
        for shard_index in range(dataset.num_shards):
            for sample_index, filename in enumerate(
                dataset.list_samples_in_shard(shard_index)
            ):
                key = self.pair_key(str(filename))
                if key in locations:
                    raise ValueError(f"duplicate pair key on {side}: {key!r}")
                locations[key] = SampleLocation(
                    shard_index=shard_index,
                    sample_index=sample_index,
                    filename=str(filename),
                )
        return locations

    def _ensure_index(self) -> None:
        if self._pairs is not None:
            return

        left = self._sample_locations(self.left, "left")
        right = self._sample_locations(self.right, "right")
        self._unmatched_left = [key for key in left if key not in right]
        self._unmatched_right = [key for key in right if key not in left]

        if self.strict and (self._unmatched_left or self._unmatched_right):
            left_example = self._unmatched_left[:3]
            right_example = self._unmatched_right[:3]
            raise ValueError(
                "paired datasets do not have identical keys: "
                f"left_only={len(self._unmatched_left)} {left_example!r}, "
                f"right_only={len(self._unmatched_right)} {right_example!r}"
            )

        self._pairs = [
            SamplePair(key=key, left=location, right=right[key])
            for key, location in left.items()
            if key in right
        ]

    @property
    def num_pairs(self) -> int:
        self._ensure_index()
        return len(self._pairs or ())

    @property
    def unmatched_left(self) -> List[str]:
        self._ensure_index()
        return list(self._unmatched_left or ())

    @property
    def unmatched_right(self) -> List[str]:
        self._ensure_index()
        return list(self._unmatched_right or ())

    def __len__(self) -> int:
        return self.num_pairs

    def get_pair(self, index: int) -> SamplePair:
        self._ensure_index()
        assert self._pairs is not None
        return self._pairs[index]

    def list_pairs(self, start: int = 0, end: Optional[int] = None) -> List[SamplePair]:
        """Return a stable slice of pairs in left-dataset order."""
        self._ensure_index()
        assert self._pairs is not None
        return self._pairs[start:end]


class PairedTarDataLoader:
    """Load joined samples without changing :class:`TarDataLoader` semantics."""

    def __init__(self, dataset: PairedDataset, **loader_kwargs: Any) -> None:
        self.dataset = dataset
        self.left_loader = TarDataLoader(dataset.left, **loader_kwargs)
        self.right_loader = TarDataLoader(dataset.right, **loader_kwargs)

    def __len__(self) -> int:
        return len(self.dataset)

    def load_pair(self, index: int) -> LoadedSamplePair:
        pair = self.dataset.get_pair(index)
        return LoadedSamplePair(
            key=pair.key,
            left=self.left_loader.load_sample(
                pair.left.shard_index, pair.left.sample_index
            ),
            right=self.right_loader.load_sample(
                pair.right.shard_index, pair.right.sample_index
            ),
        )

    def iter_pairs(
        self, start: int = 0, end: Optional[int] = None
    ) -> Iterator[LoadedSamplePair]:
        stop = len(self) if end is None else min(end, len(self))
        for index in range(start, stop):
            yield self.load_pair(index)


def _is_json_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".json"


def _sample_lookup_keys(path: str) -> List[str]:
    path_obj = Path(path)
    stem_path = str(path_obj.with_suffix(""))
    keys = [path, path_obj.name, stem_path, path_obj.stem]
    return list(dict.fromkeys(str(key) for key in keys if key))


def _normalize_captions(value: Any) -> OptionalCaptionValue:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        captions = [str(item) for item in value if item is not None and str(item)]
        return captions or None
    return str(value)


def apply_captions_to_metadata(
    metadata: MutableMapping[str, Any],
    captions_by_sample: Mapping[str, OptionalCaptionValue],
) -> int:
    """Attach captions to a webshart metadata mapping in-place.

    Captions are stored under the canonical plural ``captions`` key and may be a
    single string or a list of strings. Existing singular ``caption`` keys are
    removed from updated sample entries.
    """
    files = metadata.get("files")
    if not isinstance(files, (dict, list)):
        raise ValueError("webshart metadata must contain a 'files' dict or list")

    normalized: Dict[str, CaptionValue] = {}
    for sample, value in captions_by_sample.items():
        captions = _normalize_captions(value)
        if captions is None:
            continue
        for key in _sample_lookup_keys(str(sample)):
            normalized[key] = captions

    updated = 0

    if isinstance(files, dict):
        iterator = files.items()
    else:
        iterator = (
            (entry.get("path") or entry.get("filename") or entry.get("fname"), entry)
            for entry in files
            if isinstance(entry, dict)
        )

    for path, entry in iterator:
        if not path or not isinstance(entry, dict) or _is_json_path(str(path)):
            continue

        captions = next(
            (
                normalized[key]
                for key in _sample_lookup_keys(str(path))
                if key in normalized
            ),
            None,
        )
        if captions is None:
            continue

        entry.pop("caption", None)
        entry["captions"] = captions
        updated += 1

    return updated


def write_captions_to_metadata(
    metadata_path: Union[str, Path],
    captions_by_sample: Mapping[str, OptionalCaptionValue],
    output_path: Optional[Union[str, Path]] = None,
) -> int:
    """Write captions into a webshart shard metadata JSON file.

    Args:
        metadata_path: Existing webshart metadata JSON file to read.
        captions_by_sample: Mapping from sample path/stem to caption string or list.
        output_path: Optional destination JSON file. Defaults to updating
            ``metadata_path`` in place.

    Returns:
        Number of sample entries updated.
    """
    metadata_path = Path(metadata_path)
    destination = Path(output_path) if output_path is not None else metadata_path

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)

