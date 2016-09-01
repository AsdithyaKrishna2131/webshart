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
