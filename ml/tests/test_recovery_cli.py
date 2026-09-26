"""Check the public resume command against verified synthetic splits."""

import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def test_train_cli_resumes_and_refuses_accidental_overwrite(tmp_path):
    sys.path.insert(0, str(ROOT / "ml/scripts"))
    try:
        from audit_existing_splits import audit_splits
    finally:
        sys.path.pop(0)
    data = tmp_path / "splits"
    data.mkdir()
    for split_index, split in enumerate(("train", "validation", "test")):
        rows = []
        for index, label in enumerate(("A", "B")):
            image = data / f"{split}-{label}.png"
            Image.new("RGB", (32, 32), (20 + split_index * 30, 40 + index * 30, 90)).save(image)
            rows.append({"path": str(image), "label": label})
        (data / f"{split}.json").write_text(json.dumps(rows))
    (data / "manifest.json").write_text(json.dumps(audit_splits(data)))
    config = tmp_path / "config.json"
    config.write_text(json.dumps(dict(architecture="baseline_cnn", pretrained=False, seed=7,
                                     epochs=1, image_size=32, batch_size=2, learning_rate=0.001,
                                     num_workers=0, early_stopping_patience=10)))
    best = tmp_path / "best.pt"
    command = [sys.executable, str(ROOT / "ml/scripts/train.py"), "--config", str(config),
               "--data-dir", str(data), "--checkpoint", str(best)]
    environment = {**os.environ, "CUDA_VISIBLE_DEVICES": "-1", "PYTORCH_NVML_BASED_CUDA_CHECK": "0"}
    first = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert first.returncode == 0, first.stderr
    duplicate = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert duplicate.returncode != 0
    assert "Training output already exists" in duplicate.stderr
    resumed = subprocess.run(command + ["--resume", str(tmp_path / "best.latest.pt"), "--epochs", "2"],
                             cwd=ROOT, env=environment, capture_output=True, text=True, timeout=60)
    assert resumed.returncode == 0, resumed.stderr
    assert "Resuming after completed epoch 1" in resumed.stdout
    assert [row["epoch"] for row in json.loads((tmp_path / "best.history.json").read_text())] == [1, 2]
