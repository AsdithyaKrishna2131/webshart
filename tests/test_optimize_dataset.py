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
