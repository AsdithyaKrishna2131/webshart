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


