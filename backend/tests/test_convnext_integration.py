"""Exercise the trained architecture through registration, activation, and inference."""

from io import BytesIO
from pathlib import Path

import torch
from bone_marrow_vision.models import build_model
from PIL import Image
from pydantic import SecretStr
from test_auth import _login, _register

from app.core.config import settings
from app.inference import _load_model_cached


def test_convnext_registration_activation_prediction_and_gradcam(client, workspace_tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    for setting, directory in (("model_dir", "models"), ("config_dir", "configs"),
                               ("upload_dir", "uploads"), ("inference_temp_dir", "inputs")):
        monkeypatch.setattr(settings, setting, Path(directory))
        (workspace_tmp_path / directory).mkdir()
    monkeypatch.setattr(settings, "admin_bootstrap_token", SecretStr("convnext-test-bootstrap"))
    _register(client, "convnext-admin@example.com", bootstrap_token="convnext-test-bootstrap")
    headers = {"Authorization": f"Bearer {_login(client, 'convnext-admin@example.com')}"}
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = build_model("convnext_tiny", 3, pretrained=False)
        torch.save({"architecture": "convnext_tiny", "class_names": ["A", "B", "C"],
                    "image_size": 32, "model_state": model.state_dict()},
                   workspace_tmp_path / "models/convnext.pt")
        (workspace_tmp_path / "configs/convnext.yaml").write_text("architecture: convnext_tiny\n")
        response = client.post("/api/v1/models/register", headers=headers, json={
            "name": "ConvNeXt", "version": "test", "architecture": "convnext_tiny",
            "checkpoint_path": "models/convnext.pt", "config_path": "configs/convnext.yaml",
        })
        assert response.status_code == 201, response.text
        registered = response.json()
        assert not registered["is_active"]
        assert client.post(f"/api/v1/models/{registered['id']}/activate", headers=headers).status_code == 200
        image = BytesIO()
        Image.new("RGB", (48, 40), (90, 40, 140)).save(image, format="PNG")
        result = client.post("/api/v1/predictions", headers=headers,
                             files={"file": ("cell.png", image.getvalue(), "image/png")})
        assert result.status_code == 201, result.text
        prediction = result.json()
        assert prediction["model"]["architecture"] == "convnext_tiny"
        assert prediction["predicted_class"] in {"A", "B", "C"}
        assert 0 <= prediction["confidence"] <= 1
        media = client.get(prediction["gradcam_image_path"], headers=headers)
        assert media.status_code == 200
        with Image.open(BytesIO(media.content)) as overlay:
            assert overlay.size == (48, 40)
            overlay.verify()
        assert client.get(prediction["gradcam_image_path"]).status_code == 401
        assert not list((workspace_tmp_path / "inputs").iterdir())
    finally:
        _load_model_cached.cache_clear()
        torch.set_num_threads(previous_threads)
