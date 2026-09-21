import torch
from torch.utils.data import DataLoader, Dataset

from bone_marrow_vision.training import final_checkpoint_path, load_checkpoint
from bone_marrow_vision.training.trainer import train_model


class TinyDataset(Dataset):
    def __init__(self):
        self.samples = [{"label": "A"}, {"label": "B"}] * 3
        self.class_to_index = {"A": 0, "B": 1}
        self.images = torch.arange(6 * 3 * 4 * 4, dtype=torch.float32).reshape(6, 3, 4, 4)
        self.images /= self.images.max()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.images[index], self.class_to_index[self.samples[index]["label"]]


def test_training_writes_coherent_best_and_final_checkpoints(tmp_path):
    dataset = TinyDataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 4 * 4, 2))
    config = {
        "architecture": "baseline_cnn",
        "image_size": 4,
        "batch_size": 2,
        "epochs": 2,
        "learning_rate": 0.01,
        "class_strategy": "cross_entropy",
        "early_stopping_patience": 2,
        "seed": 4,
        "checkpoint_every": 1,
    }
    best_path = tmp_path / "best.pt"
    history = train_model(
        model,
        loader,
        loader,
        config,
        best_path,
        ["A", "B"],
        provenance={"test": True},
    )
    best = load_checkpoint(best_path)
    final = load_checkpoint(final_checkpoint_path(best_path))

    assert len(history) == len(best["history"]) == len(final["history"]) == 2
    assert best["checkpoint_role"] == "best"
    assert final["checkpoint_role"] == "final"
    assert best["epoch"] == best["best_epoch"]
    assert final["epoch"] == final["training_summary"]["final_epoch"]
    assert best["provenance"] == {"test": True}
    assert "rng" in best["training_state"]
    for epoch in (1, 2):
        snapshot = load_checkpoint(tmp_path / f"epoch_{epoch:03d}.pt")
        assert snapshot["epoch"] == epoch
        assert snapshot["checkpoint_role"] == "epoch"
        assert len(snapshot["history"]) == epoch
    assert (tmp_path / "best.history.json").is_file()
    assert all(0 <= row["validation_accuracy"] <= 1 for row in history)
    assert all(row["validation_loss"] >= 0 for row in history)


def test_load_checkpoint_keeps_legacy_tensor_only_format_compatible(tmp_path):
    path = tmp_path / "legacy.pt"
    payload = {
        "model_state": {"classifier.weight": torch.ones(2, 3)},
        "architecture": "baseline_cnn",
        "class_names": ["A", "B"],
        "image_size": 224,
        "history": [],
    }
    torch.save(payload, path)

    loaded = load_checkpoint(path)

    assert loaded["architecture"] == "baseline_cnn"
    assert loaded["class_names"] == ["A", "B"]
    assert torch.equal(loaded["model_state"]["classifier.weight"], torch.ones(2, 3))
