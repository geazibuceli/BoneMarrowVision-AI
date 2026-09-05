import json
from pathlib import Path

import pytest

from bone_marrow_vision.data.dataset import make_splits, verify_split_manifest


def _grouped_samples():
    return [
        {
            "path": f"image-{group}-{label}.png",
            "label": label,
            "patient_id": f"patient-{group}",
            "sha256": f"{group * 2 + label_index:064x}",
        }
        for group in range(12)
        for label_index, label in enumerate(("A", "B"))
    ]


def test_group_split_is_reproducible_isolated_and_manifested(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first = make_splits(
        _grouped_samples(),
        first_dir,
        seed=7,
        validation_size=0.2,
        test_size=0.2,
        group_key="patient_id",
    )
    second = make_splits(
        _grouped_samples(),
        second_dir,
        seed=7,
        validation_size=0.2,
        test_size=0.2,
        group_key="patient_id",
    )

    assert first == second
    group_sets = [
        {sample["patient_id"] for sample in first[name]} for name in ("train", "validation", "test")
    ]
    assert group_sets[0].isdisjoint(group_sets[1])
    assert group_sets[0].isdisjoint(group_sets[2])
    assert group_sets[1].isdisjoint(group_sets[2])
    manifest = verify_split_manifest(first_dir)
    assert manifest["group_key"] == "patient_id"
    assert manifest["source_sample_count"] == 24


def test_group_split_fails_closed_when_metadata_is_incomplete(tmp_path):
    samples = _grouped_samples()
    del samples[0]["patient_id"]
    with pytest.raises(ValueError, match="missing for 1 sample"):
        make_splits(samples, tmp_path, group_key="patient_id")


def test_cross_split_content_duplicates_fail_by_default(tmp_path):
    samples = _grouped_samples()
    for sample in samples:
        sample["sha256"] = "a" * 64
    with pytest.raises(ValueError, match="Duplicate image content"):
        make_splits(
            samples,
            tmp_path,
            validation_size=0.2,
            test_size=0.2,
            group_key="patient_id",
        )


def test_same_split_conflicting_duplicate_also_fails_or_is_explicitly_reported(tmp_path):
    samples = _grouped_samples()
    samples[1]["sha256"] = samples[0]["sha256"]
    with pytest.raises(ValueError, match="including 1 with conflicting labels"):
        make_splits(
            samples,
            tmp_path / "rejected",
            validation_size=0.2,
            test_size=0.2,
            group_key="patient_id",
        )

    make_splits(
        samples,
        tmp_path / "reported",
        validation_size=0.2,
        test_size=0.2,
        group_key="patient_id",
        duplicate_policy="report",
    )
    manifest = json.loads((tmp_path / "reported" / "manifest.json").read_text())
    assert manifest["source_duplicates"][0]["labels"] == ["A", "B"]


def test_split_rows_are_portable_when_absolute_inputs_are_inside_working_tree(
    tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    samples = _grouped_samples()
    for sample in samples:
        sample["path"] = str(tmp_path / sample["path"])

    make_splits(
        samples,
        tmp_path / "splits",
        validation_size=0.2,
        test_size=0.2,
        group_key="patient_id",
    )

    serialized = (tmp_path / "splits" / "train.json").read_text(encoding="utf-8")
    assert str(tmp_path) not in serialized
    assert all(not Path(row["path"]).is_absolute() for row in json.loads(serialized))


def test_manifest_detects_split_tampering(tmp_path):
    make_splits(
        _grouped_samples(),
        tmp_path,
        validation_size=0.2,
        test_size=0.2,
        group_key="patient_id",
    )
    train_path = tmp_path / "train.json"
    rows = json.loads(train_path.read_text(encoding="utf-8"))
    train_path.write_text(json.dumps(rows[:-1]), encoding="utf-8")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        verify_split_manifest(tmp_path)
