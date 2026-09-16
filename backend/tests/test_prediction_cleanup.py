from io import BytesIO
from pathlib import Path

from PIL import Image

from app.core.config import settings
from app.models import MLModel
from app.routers import predictions as prediction_router


def _png_bytes() -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color="white").save(output, format="PNG")
    return output.getvalue()


def test_successful_prediction_removes_original_input(
    client,
    session_factory,
    workspace_tmp_path: Path,
    monkeypatch,
):
    register = client.post(
        "/api/v1/auth/register",
        json={
            "email": "prediction@example.com",
            "password": "secure-test-password",
            "full_name": "Prediction User",
        },
    )
    assert register.status_code == 201
    login = client.post(
        "/api/v1/auth/login",
        data={"username": "prediction@example.com", "password": "secure-test-password"},
    )
    token = login.json()["access_token"]

    with session_factory() as database:
        database.add(
            MLModel(
                name="Active model",
                version="1",
                architecture="baseline_cnn",
                checkpoint_path="models/model.pt",
                config_path="ml/configs/baseline.yaml",
                metrics_json={},
                is_active=True,
            )
        )
        database.commit()

    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    monkeypatch.setattr(settings, "upload_dir", Path("uploads"))
    monkeypatch.setattr(settings, "inference_temp_dir", workspace_tmp_path / "temporary-inputs")
    captured_input: dict[str, Path] = {}

    def fake_predict(image_path: Path, _checkpoint_path: str, output_path: Path, threshold: float, **_kwargs):
        captured_input["path"] = image_path
        assert image_path.is_file()
        assert 0 <= threshold <= 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"gradcam")
        return {
            "predicted_class": "class-a",
            "confidence": 0.95,
            "top_predictions_json": [{"class": "class-a", "probability": 0.95}],
            "low_confidence": False,
        }

    monkeypatch.setattr(prediction_router, "predict", fake_predict)
    response = client.post(
        "/api/v1/predictions",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": ("cell.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 201
    assert not captured_input["path"].exists()
    assert list((workspace_tmp_path / "temporary-inputs").iterdir()) == []
    assert response.json()["gradcam_image_path"].endswith("/gradcam")
    saved = response.json()
    headers = {"Authorization": f"Bearer {token}"}
    history = client.get("/api/v1/predictions?limit=26&offset=0", headers=headers)
    assert history.status_code == 200
    assert history.json() == [saved]
    detail = client.get(f"/api/v1/predictions/{saved['id']}", headers=headers)
    assert detail.status_code == 200
    assert detail.json() == saved
    media = client.get(saved["gradcam_image_path"], headers=headers)
    assert media.status_code == 200
    assert media.content == b"gradcam"
