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
                "offset": 102912,
                "length": 98304,
                "sha256": "cafebabe" * 8,
            },
            {
                "path": "image_003.webp",
                "offset": 201216,
                "length": 110592,
                "sha256": "badc0de5" * 8,
            },
        ],
    }


def test_metadata_hashmap_format():
    """Test loading metadata in HashMap format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with HashMap metadata
        metadata = create_hashmap_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576

        # Check file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 3
        assert "image_001.webp" in files
        assert "image_002.webp" in files
        assert "image_003.webp" in files


def test_metadata_vec_format():
    """Test loading metadata in Vec/Array format."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dataset with Vec metadata
        metadata = create_vec_metadata()
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        # Discover and load metadata
        dataset = webshart.discover_dataset(tmpdir)

        # Force metadata loading
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 3
        assert shard_info["size"] == 1048576

        # Check file listing
        files = dataset.list_files_in_shard(0)
        assert len(files) == 3
        # Files should be accessible even in Vec format
        assert any("image_001" in f for f in files)


def test_lazy_metadata_loading():
    """Test that metadata is loaded lazily."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create multiple shards
        for i in range(3):
            metadata = {
                "path": f"data-{i:04d}.tar",
                "filesize": 1048576 * (i + 1),
                "files": {
                    f"file_{j}.webp": {"offset": j * 1024, "length": 1024}
                    for j in range(10)
                },
            }
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        # Discover dataset
        dataset = webshart.discover_dataset(tmpdir)

        # Quick stats should not load metadata
        size, files = dataset.quick_stats()
        assert size is None  # Local datasets don't have cached stats

        # Accessing specific shard should load only that metadata
        shard_info = dataset.get_shard_info(1)
        assert shard_info["num_files"] == 10

        # total_files should load all metadata
        total = dataset.total_files
        assert total == 30  # 3 shards * 10 files each


def test_metadata_with_missing_fields():
    """Test handling metadata with optional fields missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Minimal metadata
        metadata = {
            "filesize": 1024,
            "files": {"test.webp": {"offset": 0, "length": 512}},
        }
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)
        shard_info = dataset.get_shard_info(0)

        assert shard_info["num_files"] == 1
        # hash/hash_lfs are optional and may not be in the info


def test_invalid_metadata_format():
    """Test error handling for invalid metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Invalid metadata (missing required fields)
        metadata = {"some_field": "value"}
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        # Should fail when trying to access the shard
        with pytest.raises(Exception):
            dataset.get_shard_info(0)


def test_file_info_aliases():
    """Test that metadata handles field aliases (size vs length)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Use 'size' instead of 'length' (should be handled by alias)
        metadata = {
            "path": "test.tar",
            "filesize": 2048,
            "files": {
                "file1.webp": {
                    "offset": 0,
                    "size": 1024,  # Using 'size' instead of 'length'
                },
                "file2.webp": {"offset": 1024, "length": 1024},  # Using 'length'
            },
        }
        json_path = Path(tmpdir) / "data-0000.json"
        with open(json_path, "w") as f:
            json.dump(metadata, f)
        tar_path = Path(tmpdir) / "data-0000.tar"
        tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)
        files = dataset.list_files_in_shard(0)
        assert len(files) == 2


def test_batch_metadata_loading():
    """Test batch loading of metadata."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create 5 shards
        for i in range(5):
            metadata = create_hashmap_metadata()
            metadata["path"] = f"data-{i:04d}.tar"
            json_path = Path(tmpdir) / f"data-{i:04d}.json"
            with open(json_path, "w") as f:
                json.dump(metadata, f)
            tar_path = Path(tmpdir) / f"data-{i:04d}.tar"
            tar_path.touch()

        dataset = webshart.discover_dataset(tmpdir)

        # Use batch operations to load metadata
        batch_ops = webshart.BatchOperations()
        results = batch_ops.load_metadata_batch(dataset, [0, 2, 4])

        # Check all requested metadata was loaded
        assert len(results) == 3
        for result in results:
            assert result is True  # All should succeed

        # Verify metadata is actually loaded
        for idx in [0, 2, 4]:
            info = dataset.get_shard_info(idx)
            assert info["num_files"] == 3


def test_paired_json_sidecar_metadata_and_retrieval():
    """Test sample metadata and JSON sidecar retrieval for image/json webdataset pairs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "data-0000.tar"
        sample_json = {
            "caption": "a small test image",
            "text": "a small test image",
            "captions": ["a small test image", "alternate view", "alternate view"],
            "tags": ["test", "captioned"],
        }

        with tarfile.open(tar_path, "w") as tar:
            image_bytes = b"not really an image"
            image_info = tarfile.TarInfo("sample.webp")
            image_info.size = len(image_bytes)
            tar.addfile(image_info, BytesIO(image_bytes))

            json_bytes = json.dumps(sample_json).encode("utf-8")
            json_info = tarfile.TarInfo("sample.json")
            json_info.size = len(json_bytes)
            tar.addfile(json_info, BytesIO(json_bytes))

        extractor = webshart.MetadataExtractor()
        extractor.extract_metadata(
            source=tmpdir,
            destination=tmpdir,
            max_workers=1,
        )

        dataset = webshart.discover_dataset(tmpdir)
        shard_info = dataset.get_shard_info(0)
        assert shard_info["num_files"] == 2
        assert shard_info["num_samples"] == 1
        assert dataset.list_files_in_shard(0) == ["sample.json", "sample.webp"]
        assert dataset.list_samples_in_shard(0) == ["sample.webp"]

        reader = dataset.open_shard(0)
        assert reader.num_files == 2
        assert reader.num_samples == 1
        assert reader.sample_filenames() == ["sample.webp"]
        assert bytes(reader.read_sample(0)) == image_bytes
        assert json.loads(bytes(reader.read_sample_json(0))) == sample_json

        loader = webshart.TarDataLoader(dataset, load_file_data=False)
        shard_metadata = loader.get_metadata(0)
        assert shard_metadata["sample.webp"]["captions"] == [
            "a small test image",
            "alternate view",
        ]
        assert "caption" not in shard_metadata["sample.webp"]
        assert shard_metadata["sample.webp"]["json_path"] == "sample.json"
        assert shard_metadata["sample.webp"]["json_metadata"] == sample_json

        entry = loader.load_sample(0, 0)
        assert entry.path == "sample.webp"
        assert entry.caption == "a small test image"
        assert entry.captions == ["a small test image", "alternate view"]
        assert entry.metadata["captions"] == ["a small test image", "alternate view"]
        assert entry.metadata["json_path"] == "sample.json"
        assert json.loads(bytes(entry.json_data)) == sample_json


def test_txt_caption_probe_retrieval_and_metadata_export():
    """Test `.txt` sidecars are logical captions rather than separate samples."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        tar_path = root / "data-0000.tar"
        image_bytes = b"not really an image"
        caption = "A caption from a text sidecar."

        with tarfile.open(tar_path, "w") as tar:
            image_info = tarfile.TarInfo("nested/sample.webp")
            image_info.size = len(image_bytes)
            tar.addfile(image_info, BytesIO(image_bytes))

            caption_bytes = f"  {caption}\n".encode()
            caption_info = tarfile.TarInfo("nested/sample.txt")
            caption_info.size = len(caption_bytes)
            tar.addfile(caption_info, BytesIO(caption_bytes))

        webshart.MetadataExtractor().extract_metadata(
            source=tmpdir,
            destination=tmpdir,
            max_workers=1,
        )

        dataset = webshart.discover_dataset(tmpdir)
        assert dataset.list_files_in_shard(0) == [
            "nested/sample.txt",
            "nested/sample.webp",
        ]
        assert dataset.list_samples_in_shard(0) == ["nested/sample.webp"]
        assert dataset.probe_caption_layout() == {
            "layout": "txt_sidecar",
            "shards_scanned": 1,
            "total_shards": 1,
            "complete": True,
            "samples": 1,
            "captioned_samples": 1,
            "uncaptioned_samples": 0,
            "embedded_samples": 0,
            "json_sidecar_samples": 0,
            "txt_sidecar_samples": 1,
        }

        loader = webshart.TarDataLoader(dataset, load_file_data=False)
        assert loader.load_caption(0, 0) == caption
        entry = loader.load_sample(0, 0)
        assert entry.path == "nested/sample.webp"
        assert entry.caption == caption
        assert entry.captions == caption

        export_dir = root / "caption-metadata"
        result = loader.coalesce_caption_metadata(str(export_dir))
        assert result["shards"] == 1
        assert result["captioned_samples"] == 1
        assert result["coalesced_samples"] == 1
        assert result["files"] == [str(export_dir / "data-0000.json")]
        exported = json.loads((export_dir / "data-0000.json").read_text())
        assert exported["files"]["nested/sample.webp"]["captions"] == caption
        assert "captions" not in exported["files"]["nested/sample.txt"]


def test_caption_coalescing_can_persist_to_metadata_cache():
    """Test an enabled metadata cache is the default coalescing destination."""
    with tempfile.TemporaryDirectory() as tmpdir, tempfile.TemporaryDirectory() as cache:
        tar_path = Path(tmpdir) / "data-0000.tar"
        with tarfile.open(tar_path, "w") as tar:
            for filename, data in [
                ("sample.webp", b"image"),
                ("sample.txt", b"cached caption"),
            ]:
                info = tarfile.TarInfo(filename)
                info.size = len(data)
                tar.addfile(info, BytesIO(data))

        webshart.MetadataExtractor().extract_metadata(tmpdir, tmpdir, max_workers=1)
        dataset = webshart.discover_dataset(tmpdir)
        dataset.enable_metadata_cache(cache, init_shard_count=0)
        loader = webshart.TarDataLoader(dataset, load_file_data=False)

        result = loader.coalesce_caption_metadata()

        assert result["coalesced_samples"] == 1
        cached_path = Path(result["files"][0])
        assert cached_path.is_file()
        cached = json.loads(cached_path.read_text())
        assert cached["files"]["sample.webp"]["captions"] == "cached caption"


def test_txt_payload_with_json_sidecar_remains_a_logical_sample():
    """A `.txt` member is only a caption sidecar when another payload shares its stem."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tar_path = Path(tmpdir) / "data-0000.tar"
        with tarfile.open(tar_path, "w") as tar:
            for filename, data in [
