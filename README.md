<img width="1530" height="492" alt="image" src="https://github.com/user-attachments/assets/ebf0d101-eae7-4908-bb73-a264bf89a479" />

Fast dataloader and conversion utility for webdataset tar shards. Rust core with Python bindings.

Built for streaming large video and image datasets, but handles any byte data.

## Install

```bash
pip install webshart
```

## What is this?

Webshart is a fast reader for webdataset tar files with separate JSON index files. This format enables random access to any file in the dataset without downloading the entire archive.

**The indexed format** provides massive performance benefits:

- **Random access**: Jump to any file instantly
- **Selective downloads**: Only fetch the files you need
- **True parallelism**: Read from multiple shards simultaneously
- **Cloud-optimized**: Works efficiently with HTTP range requests
- **Aspect bucketing**: Optionally include image geometry hints `width`, `height` and `aspect` for the ability to bucket images by shape
- **Logical sample APIs**: Treat `image.ext` + `image.json` pairs as one sample while still allowing raw file access
- **Caption metadata**: Store captions in shard metadata under the plural `captions` key as either a string or a list of strings
- **Custom DataLoader**: Includes state dict methods on the DataLoader so that you can resume training deterministically
- **Rate-limit friendly**: Local caching allows high-frequency random seeking without encountering storage provider rate limits
- **Instant start-up** with pre-sorted aspect buckets

**Growing ecosystem**: While not all datasets use this format yet, you can easily create indices for any tar-based dataset (see below).

## Quick Start

```python
import webshart

# Find your dataset
dataset = webshart.discover_dataset(
    source="laion/conceptual-captions-12m-webdataset",
    # we're able to upload metadata separately so that we reduce load on huggingface infra.
    metadata="webshart/conceptual-captions-12m-webdataset-metadata",
)
print(f"Found {dataset.num_shards} shards")

loader = webshart.TarDataLoader(dataset)

# File-oriented access is still available.
files = dataset.list_files_in_shard(0)

# Sample-oriented access skips paired JSON sidecars.
samples = dataset.list_samples_in_shard(0)
entry = loader.load_sample(0, 0)
print(entry.path, entry.captions, entry.json_metadata)
```

### Paired datasets

Two datasets can remain independently loadable while also exposing an opt-in
join by logical sample key. This works especially well for preference,
reference, and slider-training data stored in two subfolders of one repository:

```python
paired = webshart.discover_paired_dataset(
    "webshart/suno-various-94k",
    left_subfolder="original",
    right_subfolder="covers",
)

print(paired.num_pairs)
print(paired.get_pair(0))

loader = webshart.PairedTarDataLoader(paired)
sample = loader.load_pair(0)
print(sample.key, sample.left, sample.right)
```

The normal contract is unchanged: calling `discover_dataset(...,
subfolder="original")` or `subfolder="covers"` returns a standalone dataset.
Pair indexing is lazy, preserves left-dataset order, and validates identical key
sets by default. Pass `strict=False` to use only the intersection and inspect
`unmatched_left` / `unmatched_right`.

`max_file_size` is a visibility limit for loader APIs. Files larger than the
configured limit are omitted from iteration, batches, direct sample loading,
and aspect buckets instead of being returned with empty data. Direct
`load_sample()` calls return `None` for an oversized sample. The loader's
`list_samples_in_shard()` returns dictionaries containing `sample_idx` and
`filename`, so filtered listings retain the stable index required by
`load_sample()`.

## Common Patterns

For real-world, working examples:

- [Use as a DataLoader](/examples/dataloader.py)
- [Retrieve data subset/range](/examples/retrieve_range.py)
- [Get dataset statistics without downloading](/examples/dataset_stats.py)
- [List aspect buckets](/examples/aspect_bucketing.py)
- [Write captions into metadata](/examples/write_captions_to_metadata.py)

## Creating Indices for / Converting Existing Datasets

Any tar-based webdataset can benefit from indexing! Webshart includes tools to generate indices:

A command-line tool that auto-discovers tars to process:

```bash
% webshart extract-metadata \
    --source laion/conceptual-captions-12m-webdataset \
    --destination laion_output/ \
    --checkpoint-dir ./laion_output/checkpoints \
    --max-workers 2 \
    --include-image-geometry
```

Or, if you prefer/require direct-integration to an existing Python application, [use the API](/examples/metadata_extractor.py)

### Uploading Indices to HuggingFace
