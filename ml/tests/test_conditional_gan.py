import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from bone_marrow_vision.data.dataset import sha256_file, verify_split_manifest
from bone_marrow_vision.training.conditional_gan import (
    augmentation_plan,
    generate_samples,
    train_gan,
)

spec = importlib.util.spec_from_file_location(
    "gan_experiment", Path(__file__).parents[1] / "scripts" / "run_gan_experiment.py"
)
experiment = importlib.util.module_from_spec(spec)
spec.loader.exec_module(experiment)


def test_paired_comparison_reports_regression_and_rejects_different_holdouts():
    control = {"test_split_sha256": "same", "test_accuracy": 0.86,
               "test_macro_precision": 0.75, "test_macro_f1": 0.73, "test_loss": 0.58}
    augmented = {**control, "test_accuracy": 0.87, "test_macro_f1": 0.72}
    comparison = experiment.paired_comparison(augmented, control)
    assert comparison["comparison_type"] == "matched_real_only_control"
    assert comparison["delta_gan_minus_control"]["test_macro_f1"] == pytest.approx(-0.01)
    with pytest.raises(ValueError, match="different test splits"):
        experiment.paired_comparison({**augmented, "test_split_sha256": "other"}, control)


def test_matched_holdouts_reject_validation_changes(tmp_path):
    real, augmented = tmp_path / "real", tmp_path / "augmented"
    real.mkdir()
    augmented.mkdir()
    for name in ("validation", "test"):
        (real / f"{name}.json").write_text("[]")
        (augmented / f"{name}.json").write_text("[]")
    experiment.verify_matched_holdouts(real, augmented)
    (augmented / "validation.json").write_text('[{"label":"changed"}]')
    with pytest.raises(RuntimeError, match="Held-out split changed"):
        experiment.verify_matched_holdouts(real, augmented)


def source_splits(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    random = np.random.default_rng(42)
    for split in ("train", "validation", "test"):
        rows = []
        for index, label in enumerate(["A", "A", "B", "B", "B", "B"]):
            image_path = source / f"{split}_{index}.png"
            Image.fromarray(random.integers(0, 256, (16, 16, 3), dtype=np.uint8)).save(image_path)
            rows.append({"path": str(image_path), "label": label})
        (source / f"{split}.json").write_text(json.dumps(rows), encoding="utf-8")
    return source


def test_gan_train_generate_and_preserve_real_holdouts(tmp_path):
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        source = source_splits(tmp_path)
        real = tmp_path / "real"
        splits = experiment.prepare_real_splits(source, real)
        config = {"seed": 42, "epochs": 1, "steps_per_epoch": 1, "batch_size": 2,
                  "latent_dim": 4, "width": 2, "learning_rate": 0.0002,
                  "num_workers": 0, "checkpoint_every": 1, "device": "cpu"}
        checkpoint = train_gan(real, tmp_path / "gan", config)
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        assert payload["epoch"] == 1
        assert np.isfinite(payload["history"][0]["generator_loss"])
        synthetic = generate_samples(checkpoint, real, tmp_path / "synthetic", 13)
        assert len(synthetic) == 2
        assert all(row["label"] == "A" and row["is_synthetic"] for row in synthetic)
        assert all(Image.open(row["path"]).size == (128, 128) for row in synthetic)
        duplicate_run = generate_samples(checkpoint, real, tmp_path / "repeat", 13)
        assert [r["sha256"] for r in synthetic] == [r["sha256"] for r in duplicate_run]
        augmented = tmp_path / "augmented"
        experiment.write_splits(augmented, {**splits, "train": splits["train"] + synthetic}, {})
        verify_split_manifest(augmented)
        for name in ("validation", "test"):
            assert sha256_file(real / f"{name}.json") == sha256_file(augmented / f"{name}.json")
        (real / "train.json").write_text("[]", encoding="utf-8")
        with pytest.raises(ValueError, match="SHA-256 mismatch"):
            generate_samples(checkpoint, real, tmp_path / "tampered", 13)
    finally:
        torch.set_num_threads(original_threads)


@pytest.mark.parametrize("different_encoding", [False, True])
def test_audit_rejects_cross_split_image_leakage(tmp_path, different_encoding):
    source = source_splits(tmp_path)
    train = json.loads((source / "train.json").read_text())
    validation = json.loads((source / "validation.json").read_text())
    path = validation[0]["path"]
    if different_encoding:
        path = str(source / "duplicate.bmp")
        with Image.open(validation[0]["path"]) as image:
            image.save(path)
    train[0]["path"] = path
    (source / "train.json").write_text(json.dumps(train))
    with pytest.raises(ValueError, match="cross source splits"):
        experiment.prepare_real_splits(source, tmp_path / "real")


def test_augmentation_is_capped_for_rare_classes():
    assert augmentation_plan({"rare": 6, "medium": 300, "common": 1000}, 500) == {
        "rare": 6, "medium": 200, "common": 0,
    }


def test_gan_interrupted_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    from bone_marrow_vision.training import conditional_gan
    original_threads = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        real = tmp_path / "real"
        experiment.prepare_real_splits(source_splits(tmp_path), real)
        config = {"seed": 42, "epochs": 2, "steps_per_epoch": 1, "batch_size": 2,
                  "latent_dim": 4, "width": 2, "learning_rate": 0.0002,
                  "num_workers": 0, "checkpoint_every": 1, "device": "cpu"}
        full = train_gan(real, tmp_path / "full", config)
        save = conditional_gan.atomic_torch_save

        def interrupted(payload, path):
            save(payload, path)
            if path.name == "latest.pt" and payload["epoch"] == 1:
                raise RuntimeError("simulated interruption")
        monkeypatch.setattr(conditional_gan, "atomic_torch_save", interrupted)
        with pytest.raises(RuntimeError, match="simulated interruption"):
            train_gan(real, tmp_path / "resumed", config)
        monkeypatch.setattr(conditional_gan, "atomic_torch_save", save)
        resumed = train_gan(real, tmp_path / "resumed", config, resume=True)
        expected = torch.load(full, weights_only=True, map_location="cpu")
        actual = torch.load(resumed, weights_only=True, map_location="cpu")
        assert actual["history"] == expected["history"]
        for model in ("generator", "discriminator"):
            assert all(torch.equal(value, actual[model][key]) for key, value in expected[model].items())
        with pytest.raises(ValueError, match="inputs changed"):
            train_gan(real, tmp_path / "resumed", {**config, "seed": 99}, resume=True)
    finally:
        torch.set_num_threads(original_threads)


def test_quarantine_excludes_entire_conflicting_group_without_editing_source(tmp_path):
    source = source_splits(tmp_path)
    train_path = source / "train.json"
    train = json.loads(train_path.read_text())
    validation = json.loads((source / "validation.json").read_text())
    train[2]["path"] = validation[0]["path"]
    train_path.write_text(json.dumps(train))
    before = sha256_file(train_path)
    real = tmp_path / "real"
    splits = experiment.prepare_real_splits(source, real, duplicate_policy="quarantine")
    assert len(splits["train"]) == 5
    assert len(splits["validation"]) == 5
    assert len(splits["test"]) == 6
    assert all(row["path"] != validation[0]["path"] for rows in splits.values() for row in rows)
    assert sha256_file(train_path) == before
    assert len(json.loads((tmp_path / "source_audit.json").read_text())["conflicting_label_groups"]) == 1


@pytest.mark.parametrize("fail_training", [False, True])
def test_runner_matches_classifier_arms_and_records_completion(tmp_path, monkeypatch, fail_training):
    source = source_splits(tmp_path)
    output = tmp_path / "paired"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(json.dumps({
        "seed": 42, "gan": {"seed": 42, "epochs": 1},
        "classifier": {"seed": 42, "epochs": 2, "early_stopping_patience": 3},
    }))
    monkeypatch.chdir(Path(__file__).parents[2])
    monkeypatch.setattr(experiment.sys, "argv", ["run_gan_experiment.py", "--config", str(config_path),
                                              "--data-dir", str(source), "--output", str(output),
                                              "--seed", "43"])
    monkeypatch.setattr(experiment.torch.cuda, "is_available", lambda: True)
    events = []
    configs = []

    def fake_gan(real_dir, destination, config, *, resume=False):
        assert config["seed"] == 43
        destination.mkdir()
        (destination / "latest.pt").write_bytes(b"checkpoint")
        return destination / "latest.pt"

    def fake_generation(checkpoint, real_dir, destination, seed, *, resume=False):
        assert seed == 43
        destination.mkdir()
        image_path = destination / "synthetic.png"
        Image.new("RGB", (16, 16), (12, 34, 56)).save(image_path)
        rows = [{"path": str(image_path), "label": "A", "is_synthetic": True,
                 "sha256": sha256_file(image_path)}]
        experiment.write_json(destination / "manifest.json", {"seed": seed, "samples": rows})
        return rows

    def fake_training(config, data, checkpoint, export_directory, *, resume=False):
        events.append(data.name)
        configs.append(config.read_bytes())
        if fail_training:
            raise RuntimeError("Training interrupted")
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_bytes(b"classifier")
        checkpoint.with_suffix(".history.json").write_text("[]")

    def fake_report(command, check):
        assert events[:2] == ["real_splits", "augmented_splits"]
        assert check
        events.append("report")

    def fake_evaluation(checkpoint, test):
        events.append("test")
        return {"checkpoint_sha256": checkpoint.parent.name, "test_split_sha256": sha256_file(test),
                "test_accuracy": 0.8, "test_macro_precision": 0.7,
                "test_macro_f1": 0.6, "test_loss": 0.5}

    monkeypatch.setattr(experiment, "train_gan", fake_gan)
    monkeypatch.setattr(experiment, "generate_samples", fake_generation)
    monkeypatch.setattr(experiment, "train_classifier", fake_training)
    monkeypatch.setattr(experiment.subprocess, "run", fake_report)
    monkeypatch.setattr(experiment, "evaluate_checkpoint", fake_evaluation)
    if fail_training:
        with pytest.raises(RuntimeError, match="Training interrupted"):
            experiment.main()
        status = json.loads((output / "status.json").read_text())
        assert status["stage"] == "failed"
        assert status["error"] == "Training interrupted"
        assert not (output / "comparison.json").exists()
        assert events == ["real_splits"]
    else:
        experiment.main()
        assert configs[0] == configs[1]
        assert experiment.yaml.safe_load(configs[0])["seed"] == 43
        assert events == ["real_splits", "augmented_splits", "report", "report", "test", "test"]
        comparison = json.loads((output / "comparison.json").read_text())
        assert comparison["seed"] == 43
        assert comparison["delta_gan_minus_control"]["test_macro_f1"] == 0
        status = json.loads((output / "status.json").read_text())
        assert status["stage"] == "completed"
        assert status["stages"][0]["stage"] == "auditing_real_splits"
    assert status["elapsed_seconds"] >= 0
    assert status["started_at"] <= status["updated_at"]

    # The second invocation must reuse GAN generation and both completed classifiers.
    if not fail_training:
        experiment.sys.argv.append("--resume")
        prior_training = list(configs)
        original_provenance = (output / "provenance.json").read_bytes()
        experiment.main()
        assert configs == prior_training
        assert (output / "provenance.json").read_bytes() == original_provenance
        assert len(json.loads((output / "resume_provenance.json").read_text())) == 1
        assert json.loads((output / "status.json").read_text())["stage"] == "completed"
