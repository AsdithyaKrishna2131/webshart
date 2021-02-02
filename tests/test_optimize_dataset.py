"""Tests for rolling loose-dataset optimization."""

from pathlib import Path
import json
import tarfile

import webshart
import webshart.optimize as optimize_module


def _write_loose_pairs(root: Path, count: int = 5) -> None:
    for index in range(count):
        folder = root / "nested" / f"group-{index % 2}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / f"sample-{index}.jpg").write_bytes(f"payload-{index}".encode("utf-8"))
        (folder / f"sample-{index}.txt").write_text(
            f" caption {index}\n", encoding="utf-8"
        )


def _write_legacy_tar(root: Path, count: int = 3) -> None:
    root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(root / "train_0000.tar", "w") as archive:
        for index in range(count):
            payload = root / f"caption_{index}.jpg"
            payload.write_bytes(f"legacy-{index}".encode())
            archive.add(payload, arcname=f"./caption_{index}.jpg")
            payload.unlink()


def test_optimize_dataset_repackages_legacy_tars_and_resumes(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_legacy_tar(source)

    first = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
        max_shards=1,
    )
    assert first["input_layout"] == "legacy_tar"
    assert first["status"] == "running"
    assert first["next_sample_index"] == 1
    assert first["next_source_archive_index"] == 0
    assert first["next_source_member_offset"] > 0

    second = webshart.optimize_dataset(
        source,
        destination=destination,
        max_shard_size_bytes=1_500,
        include_image_geometry=False,
    )
    assert second["status"] == "complete"
    assert second["resumed"] is True
    assert second["next_sample_index"] == 3
    assert second["next_source_archive_index"] == 1
    assert second["captioned_samples"] == 3

    output = destination / "webshart"
    with tarfile.open(output / "shard-00000.tar") as archive:
        assert archive.getnames() == ["train_0000/caption_0.jpg"]
    metadata = json.loads((output / "shard-00000.json").read_text())
    entry = metadata["files"]["train_0000/caption_0.jpg"]
    assert entry["captions"] == "caption 0"

    state_text = (output / ".webshart-optimize-state.json").read_text()
    assert str(source.resolve()) not in state_text
    assert str(destination.resolve()) not in state_text


def test_optimize_dataset_skips_truncated_legacy_tar_tail(tmp_path):
    source = tmp_path / "source"
    destination = tmp_path / "output"
    _write_legacy_tar(source, count=2)
    archive_path = source / "train_0000.tar"
    with tarfile.open(archive_path) as archive:
        last = archive.getmembers()[-1]
        truncated_size = last.offset_data + last.size // 2
    with archive_path.open("r+b") as handle:
        handle.truncate(truncated_size)

    result = webshart.optimize_dataset(
        source,
        destination=destination,
        include_image_geometry=False,
    )

    assert result["status"] == "complete"
    assert result["next_sample_index"] == 1
    with tarfile.open(destination / "webshart" / "shard-00000.tar") as archive:
