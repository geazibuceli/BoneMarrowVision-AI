import json

import pytest

from app.core.config import settings
from app.experiment_artifacts import run_directory


@pytest.fixture
def run_headers(client):
    client.post("/api/v1/auth/register", json={
        "email": "runs@example.com", "password": "experiment-password", "full_name": "Run Reader",
    })
    response = client.post("/api/v1/auth/login", data={
        "username": "runs@example.com", "password": "experiment-password",
    })
    return {"Authorization": f"Bearer {response.json()['access_token']}"}


def test_local_runs_require_authentication(client):
    for path in ("/runs", "/runs/example", "/runs/example/curves"):
        assert client.get(f"/api/v1/experiments{path}").status_code == 401


def test_completed_and_partial_runs(client, run_headers, workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    run = workspace_tmp_path / "data/experiments/example"
    run.mkdir(parents=True)
    (run / "status.json").write_text('{"stage": "training_classifier"}')
    (run / "config.json").write_text('{"seed": 42}')
    response = client.get("/api/v1/experiments/runs", headers=run_headers)
    assert response.json() == [{"id": "example", "status": {"stage": "training_classifier"}}]
    detail = client.get("/api/v1/experiments/runs/example", headers=run_headers)
    assert detail.status_code == 200
    assert detail.json()["configuration"] == {"seed": 42}
    assert detail.json()["comparison"] is None
    comparison = {"gan_augmented": {"test_macro_f1": 0.72}, "limitations": "Historical only"}
    (run / "comparison.json").write_text(json.dumps(comparison))
    (run / "status.json").write_text('{"stage": "completed"}')
    detail = client.get("/api/v1/experiments/runs/example", headers=run_headers).json()
    assert detail["comparison"] == comparison
    assert detail["status"]["stage"] == "completed"
    assert client.get("/api/v1/experiments/runs/missing", headers=run_headers).status_code == 404
    assert client.get("/api/v1/experiments/runs/example/curves", headers=run_headers).status_code == 404


def test_malformed_run_does_not_hide_valid_runs(client, run_headers, workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    for name, content in (("valid", '{"stage":"completed"}'), ("broken", "{"), ("wrong", "[]")):
        run = workspace_tmp_path / "data/experiments" / name
        run.mkdir(parents=True)
        (run / "status.json").write_text(content)
    response = client.get("/api/v1/experiments/runs", headers=run_headers)
    assert [row["id"] for row in response.json()] == ["valid"]
    assert client.get("/api/v1/experiments/runs/broken", headers=run_headers).status_code == 503


@pytest.mark.parametrize("name", ["..", "../models", "a/b", "a\\b", "C:\\Windows", ".hidden"])
def test_run_paths_cannot_escape_root(name):
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        run_directory(name)
    assert error.value.status_code == 404


@pytest.mark.parametrize("content", ['[null]', '[{"epoch":1,"validation_macro_f1":NaN}]', '{}'])
def test_invalid_history_returns_recoverable_error(client, run_headers, workspace_tmp_path, monkeypatch, content):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    run = workspace_tmp_path / "data/experiments/invalid-history"
    (run / "classifier").mkdir(parents=True)
    (run / "status.json").write_text('{"stage":"completed"}')
    (run / "classifier/best.history.json").write_text(content)
    response = client.get("/api/v1/experiments/runs/invalid-history", headers=run_headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Experiment artifact is unavailable or invalid"


def test_architecture_runs_are_visible_and_sanitize_private_paths(client, run_headers, workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    monkeypatch.setattr(settings, "model_dir", "models")
    directory = workspace_tmp_path / "models/comparison"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text(json.dumps({
        "status": "completed", "steps": [{"command": ["C:/private/python.exe"]}],
        "error": "C:/private/error", "inputs": {},
    }))
    for architecture in ("resnet50", "convnext_tiny"):
        (directory / f"{architecture}.yaml").write_text(f"architecture: {architecture}\nseed: 42\n")
        (directory / architecture).mkdir()
        report = {key: 0.8 for key in ("accuracy", "balanced_accuracy", "macro_precision",
                                      "macro_recall", "macro_f1", "weighted_f1")}
        report.update(class_names=["A"], confusion_matrix=[[2]], provenance={
            "test_split": {"sha256": "a" * 64, "path": "C:/private/test.json"},
        })
        (directory / architecture / "test_report.json").write_text(json.dumps(report))
    listing = client.get("/api/v1/experiments/runs", headers=run_headers).json()
    assert listing[0]["id"] == "models~comparison"
    response = client.get("/api/v1/experiments/runs/models~comparison", headers=run_headers)
    assert response.status_code == 200
    assert "private" not in response.text
    assert response.json()["matching_test_splits"] is True
    assert len(response.json()["architectures"]) == 2
    (directory / "convnext_tiny/test_report.json").unlink()
    result = client.get("/api/v1/experiments/runs/models~comparison", headers=run_headers).json()
    assert result["matching_test_splits"] is False
    assert result["architectures"][1]["test_report"] is None
    assert client.get("/api/v1/experiments/runs/models~comparison").status_code == 401


@pytest.mark.parametrize("name", ["models~..", "models~../secret", "models~a/b", "models~C:\\Windows"])
def test_architecture_paths_cannot_escape_root(name):
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as error:
        run_directory(name)
    assert error.value.status_code == 404


@pytest.mark.parametrize("value", [".nan", ".inf", "-.inf"])
def test_non_finite_architecture_configuration_returns_recoverable_error(
    client, run_headers, workspace_tmp_path, monkeypatch, value,
):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    monkeypatch.setattr(settings, "model_dir", "models")
    directory = workspace_tmp_path / "models/invalid-config"
    directory.mkdir(parents=True)
    (directory / "status.json").write_text('{"status":"completed"}')
    (directory / "resnet50.yaml").write_text(f"architecture: resnet50\nlearning_rate: {value}\n")
    response = client.get("/api/v1/experiments/runs/models~invalid-config", headers=run_headers)
    assert response.status_code == 503
    assert response.json()["detail"] == "Experiment artifact is unavailable or invalid"
