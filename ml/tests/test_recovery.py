import copy

import pytest
import torch
from torch.utils.data import DataLoader

from bone_marrow_vision.training import (
    load_checkpoint,
    make_generator,
    seed_everything,
    seed_worker,
)
from bone_marrow_vision.training.recovery import recovery_path
from bone_marrow_vision.training.trainer import make_sampler, train_model


class TinyDataset(torch.utils.data.Dataset):
    samples = [{"label": "A"}, {"label": "B"}] * 3
    class_to_index = {"A": 0, "B": 1}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return torch.full((3, 4, 4), index / 6) + torch.rand(3, 4, 4) * 0.01, index % 2


def setup_run(workers=0, balanced=False):
    seed_everything(7)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Dropout(0.2), torch.nn.Linear(48, 2))
    sampler = make_sampler([0, 1] * 3, generator=make_generator(9)) if balanced else None
    train = DataLoader(TinyDataset(), batch_size=2, shuffle=sampler is None, sampler=sampler,
                       generator=make_generator(7), num_workers=workers, worker_init_fn=seed_worker)
    valid = DataLoader(TinyDataset(), batch_size=2, generator=make_generator(8),
                       num_workers=workers, worker_init_fn=seed_worker)
    config = dict(architecture="baseline_cnn", image_size=4, batch_size=2, epochs=4,
                  learning_rate=0.01, early_stopping_patience=10, seed=7)
    provenance = {"input_files": {"train_split": {"sha256": "original"}}}
    return model, train, valid, config, provenance


@pytest.mark.parametrize(("workers", "balanced"), [(0, False), (1, True)])
def test_recovery_matches_uninterrupted_training_after_crash(tmp_path, monkeypatch, workers, balanced):
    from bone_marrow_vision.training import trainer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, train, valid, config, provenance = setup_run(workers, balanced)
    full = tmp_path / "full.pt"
    train_model(model, train, valid, config, full, ["A", "B"], provenance=provenance)
    expected = load_checkpoint(tmp_path / "full.final.pt")
    model, train, valid, config, provenance = setup_run(workers, balanced)
    interrupted = tmp_path / "interrupted.pt"
    real_validation = trainer.validation_metrics
    calls = 0

    def crash(*args):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("Simulated interruption")
        return real_validation(*args)

    monkeypatch.setattr(trainer, "validation_metrics", crash)
    with pytest.raises(RuntimeError, match="Simulated interruption"):
        train_model(model, train, valid, config, interrupted, ["A", "B"], provenance=provenance)
    assert load_checkpoint(recovery_path(interrupted))["epoch"] == 2
    monkeypatch.setattr(trainer, "validation_metrics", real_validation)
    model, train, valid, config, provenance = setup_run(workers, balanced)
    train_model(model, train, valid, config, interrupted, ["A", "B"],
                provenance=provenance, resume=recovery_path(interrupted))
    actual = load_checkpoint(tmp_path / "interrupted.final.pt")
    assert actual["history"] == expected["history"]
    assert actual["training_summary"] == expected["training_summary"]
    assert actual["training_state"]["scheduler"] == expected["training_state"]["scheduler"]
    for key, value in actual["model_state"].items():
        assert torch.equal(value, expected["model_state"][key])
    best = load_checkpoint(interrupted)
    full_best = load_checkpoint(full)
    assert best["epoch"] == full_best["epoch"]
    for key, value in best["model_state"].items():
        assert torch.equal(value, full_best["model_state"][key])


@pytest.mark.parametrize("change", ["config", "data", "classes", "epochs", "legacy"])
def test_recovery_rejects_incompatible_state_without_overwriting(tmp_path, monkeypatch, change):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    model, train, valid, config, provenance = setup_run()
    config["epochs"] = 1
    output = tmp_path / "best.pt"
    train_model(model, train, valid, config, output, ["A", "B"], provenance=provenance)
    before = output.read_bytes()
    config = copy.deepcopy(config)
    config["epochs"] = 2
    classes = ["A", "B"]
    resume = recovery_path(output)
    if change == "config":
        config["learning_rate"] = 0.5
    elif change == "data":
        provenance["input_files"]["train_split"]["sha256"] = "changed"
    elif change == "classes":
        classes.reverse()
    elif change == "epochs":
        config["epochs"] = 1
    else:
        resume = output
    with pytest.raises(ValueError):
        train_model(model, train, valid, config, output, classes, provenance=provenance, resume=resume)
    assert output.read_bytes() == before
