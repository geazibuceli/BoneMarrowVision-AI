import json

import pytest
from PIL import Image

from bone_marrow_vision.data.dataset import discover_samples, make_splits
from bone_marrow_vision.data.inspection import inspect_dataset


def _make_small_dataset(root):
    for label in ("A", "B"):
        directory = root / label
        directory.mkdir(parents=True)
        for index in range(10):
            label_offset = 100 if label == "B" else 0
            Image.new("RGB", (4, 4), (label_offset + index * 10, 30, 60)).save(
                directory / f"{label}-{index}.png"
            )
    (root / "B" / "B-0.png").write_bytes((root / "A" / "A-0.png").read_bytes())
    (root / "A" / "broken.jpg").write_bytes(b"incomplete image")
    (root / "download.partial").write_bytes(b"partial")


def test_inspection_reports_invalid_duplicates_and_only_portable_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "dataset"
    _make_small_dataset(root)

    report = inspect_dataset(root, tmp_path / "dataset-report.json")

    assert report["image_count"] == 21
    assert report["readable_image_count"] == 20
    assert report["unreadable_images"] == ["A/broken.jpg"]
    assert report["incomplete_download_files"] == ["download.partial"]
    assert len(report["duplicate_groups"]) == 1
    assert str(tmp_path) not in json.dumps(report)
    assert report["data_directory"] == "dataset"
    assert report["image_manifest"]["path"] == "dataset-report.images.jsonl"


def test_documented_preparation_flow_fails_clearly_before_writing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    root = tmp_path / "dataset"
    _make_small_dataset(root)
    output = tmp_path / "processed"

    with pytest.raises(ValueError, match="Found 1 unreadable image"):
        discover_samples(root)
    assert not output.exists()

    samples = discover_samples(root, strict=False)
    with pytest.raises(ValueError, match="Duplicate image content"):
        make_splits(samples, output, validation_size=0.2, test_size=0.2)
    assert not output.exists()


def test_inspection_rejects_a_missing_dataset(tmp_path):
    with pytest.raises(ValueError, match="Dataset directory does not exist"):
        inspect_dataset(tmp_path / "missing", tmp_path / "report.json")
