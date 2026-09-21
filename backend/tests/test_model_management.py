from pathlib import Path

import pytest
from pydantic import SecretStr
from test_auth import _login, _register

from app.core.config import settings
from app.models import Prediction
from app.routers import models as model_router


@pytest.fixture
def registry(client, workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "admin_bootstrap_token", SecretStr("model-test-bootstrap"))
    _register(client, "model-admin@example.com", bootstrap_token="model-test-bootstrap")
    researcher = _register(client, "model-reader@example.com").json()
    admin_headers = {"Authorization": f"Bearer {_login(client, 'model-admin@example.com')}"}
    reader_headers = {"Authorization": f"Bearer {_login(client, 'model-reader@example.com')}"}
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    monkeypatch.setattr(settings, "model_dir", Path("models"))
    monkeypatch.setattr(settings, "config_dir", Path("configs"))
    (workspace_tmp_path / "models").mkdir()
    (workspace_tmp_path / "configs").mkdir()
    (workspace_tmp_path / "configs/model.yaml").write_text("architecture: baseline_cnn")
    monkeypatch.setattr(model_router, "load_model", lambda *_args: None)
    registered = []
    for version in ("1", "2"):
        (workspace_tmp_path / f"models/model-{version}.pt").write_bytes(f"fixture-{version}".encode())
        response = client.post("/api/v1/models/register", headers=admin_headers, json={
            "name": "Classifier", "version": version, "architecture": "baseline_cnn",
            "checkpoint_path": f"models/model-{version}.pt", "config_path": "configs/model.yaml",
            "metrics_json": {"macro_f1": 0.73},
        })
        assert response.status_code == 201
        assert not response.json()["is_active"]
        registered.append(response.json())
    return admin_headers, reader_headers, researcher["id"], registered


def test_activation_restore_and_prediction_model_identity(client, session_factory, registry):
    admin, reader, owner_id, models = registry
    first, second = models
    assert client.post(f"/api/v1/models/{first['id']}/activate", headers=admin).status_code == 200
    with session_factory() as database:
        prediction = Prediction(user_id=owner_id, model_id=first["id"], predicted_class="A", confidence=0.9,
                                top_predictions_json=[])
        legacy = Prediction(user_id=owner_id, predicted_class="B", confidence=0.8, top_predictions_json=[])
        database.add_all([prediction, legacy])
        database.commit()
        prediction_id, legacy_id = prediction.id, legacy.id
    assert client.post(f"/api/v1/models/{second['id']}/activate", headers=admin).status_code == 200
    detail = client.get(f"/api/v1/predictions/{prediction_id}", headers=reader).json()
    assert detail["model"] == {"id": first["id"], "name": "Classifier", "version": "1",
                               "architecture": "baseline_cnn"}
    assert client.get(f"/api/v1/predictions/{legacy_id}", headers=reader).json()["model"] is None
    assert client.get(f"/api/v1/predictions/{prediction_id}", headers=admin).status_code == 404
    history = client.get("/api/v1/predictions", headers=reader).json()
    assert next(row for row in history if row["id"] == prediction_id)["model"] == detail["model"]

    assert client.post(f"/api/v1/models/{first['id']}/activate", headers=admin).status_code == 200
    assert client.post(f"/api/v1/models/{first['id']}/activate", headers=admin).status_code == 200
    events = client.get("/api/v1/models/activation-history", headers=admin).json()
    assert [event["model_id"] for event in events] == [first["id"], second["id"], first["id"]]
    assert events[0]["model_version"] == "1"
    assert events[0]["created_at"].endswith(("Z", "+00:00"))
    page = client.get("/api/v1/models/activation-history?limit=1&offset=1", headers=admin).json()
    assert page == events[1:2]
    rows = client.get("/api/v1/models", headers=reader).json()
    assert [row["id"] for row in rows if row["is_active"]] == [first["id"]]


def test_failed_validation_keeps_active_model_and_history(client, registry, monkeypatch):
    admin, reader, _, models = registry
    first, second = models
    client.post(f"/api/v1/models/{first['id']}/activate", headers=admin)

    def broken_checkpoint(path, architecture, digest):
        assert digest == second["checkpoint_sha256"]
        raise ValueError("Checkpoint hash changed")

    monkeypatch.setattr(model_router, "load_model", broken_checkpoint)
    response = client.post(f"/api/v1/models/{second['id']}/activate", headers=admin)
    assert response.status_code == 422
    assert "active model unchanged" in response.json()["detail"]
    rows = client.get("/api/v1/models", headers=reader).json()
    assert [row["id"] for row in rows if row["is_active"]] == [first["id"]]
    assert len(client.get("/api/v1/models/activation-history", headers=admin).json()) == 1


def test_model_administration_requires_admin(client, registry):
    admin, reader, _, models = registry
    path = f"/api/v1/models/{models[0]['id']}/activate"
    assert client.post(path).status_code == 401
    assert client.post(path, headers=reader).status_code == 403
    assert client.get("/api/v1/models/activation-history").status_code == 401
    assert client.get("/api/v1/models/activation-history", headers=reader).status_code == 403
    assert client.get("/api/v1/models/activation-history?offset=-1", headers=admin).status_code == 422
    assert client.get("/api/v1/models/activation-history?limit=101", headers=admin).status_code == 422
