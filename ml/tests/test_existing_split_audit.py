import importlib.util
import json
from pathlib import Path

import pytest
from PIL import Image

from bone_marrow_vision.data.dataset import verify_split_manifest

spec = importlib.util.spec_from_file_location(
    "audit_existing_splits", Path(__file__).parents[1] / "scripts/audit_existing_splits.py"
)
audit_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit_module)


def prepare(directory, duplicate=False):
    for index, name in enumerate(("train", "validation", "test")):
        image = directory / f"{name}.png"
        Image.new("RGB", (8, 8), (0 if duplicate else index * 50, 0, 0)).save(image)
        (directory / f"{name}.json").write_text(
            json.dumps([{"path": str(image), "label": "A"}]), encoding="utf-8"
        )


def test_audit_preserves_splits_and_creates_verifiable_manifest(tmp_path):
    prepare(tmp_path)
    before = {p: p.read_bytes() for p in tmp_path.glob("*.json")}
    manifest = audit_module.audit_splits(tmp_path)
    assert not manifest["cross_split_duplicates"]
    assert manifest["group_isolation_verified"] is False
    assert all(p.read_bytes() == contents for p, contents in before.items())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    assert verify_split_manifest(tmp_path)["splits"]["train"]["sample_count"] == 1


def test_different_paths_with_identical_content_are_rejected(tmp_path):
    prepare(tmp_path, duplicate=True)
    manifest = audit_module.audit_splits(tmp_path)
    assert len(manifest["cross_split_duplicates"]) == 1
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="forbidden content duplicates"):
        verify_split_manifest(tmp_path)


def test_unreadable_image_does_not_produce_manifest(tmp_path):
    prepare(tmp_path)
    (tmp_path / "test.png").write_bytes(b"broken image")
    with pytest.raises(OSError):
        audit_module.audit_splits(tmp_path)
    assert not (tmp_path / "manifest.json").exists()


def test_conflicting_validation_labels_are_removed_only_from_copy(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    from prepare_comparison_splits import prepare_comparison

    source = tmp_path / "source"
    source.mkdir()
    prepare(source)
    # Keep both classes represented after excluding the conflicting pair.
    for index, name in enumerate(("train", "validation", "test")):
        image = source / f"{name}_b.png"
        Image.new("RGB", (8, 8), (0, index * 50 + 30, 0)).save(image)
        path = source / f"{name}.json"
        rows = json.loads(path.read_text())
        rows.append({"path": str(image), "label": "B"})
        path.write_text(json.dumps(rows))
    conflicting = source / "conflicting.png"
    Image.new("RGB", (8, 8), (255, 255, 255)).save(conflicting)
    validation = source / "validation.json"
    rows = json.loads(validation.read_text())
    copy = source / "conflicting_copy.png"
    copy.write_bytes(conflicting.read_bytes())
    rows.extend([{"path": str(conflicting), "label": "A"}, {"path": str(copy), "label": "B"}])
    validation.write_text(json.dumps(rows))
    audit = audit_module.audit_splits(source)
    assert len(audit["conflicting_labels"]) == 1
    (source / "existing_split_audit.json").write_text(json.dumps(audit))
    before = {p.name: p.read_bytes() for p in source.glob("*.json")}
    output = tmp_path / "corrected"
    prepare_comparison(source, output)
    assert {p.name: p.read_bytes() for p in source.glob("*.json")} == before
    assert (output / "train.json").read_bytes() == before["train.json"]
    assert (output / "test.json").read_bytes() == before["test.json"]
    assert len(json.loads((output / "validation.json").read_text())) == 2
    assert not verify_split_manifest(output)["conflicting_labels"]
    with pytest.raises(FileExistsError):
        prepare_comparison(source, output)
    validation.write_text("[]")
    with pytest.raises(ValueError, match="changed since audit"):
        prepare_comparison(source, tmp_path / "changed")
