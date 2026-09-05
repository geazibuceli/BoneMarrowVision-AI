from pathlib import Path

from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.models import Prediction


def _create_user_and_token(client, email: str) -> tuple[int, str]:
    register = client.post(
        "/api/v1/auth/register",
        json={"email": email, "password": "secure-test-password", "full_name": "Media User"},
    )
    assert register.status_code == 201
    login = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": "secure-test-password"},
    )
    assert login.status_code == 200
    return register.json()["id"], login.json()["access_token"]


def test_gradcam_is_private_and_scoped_to_prediction_owner(
    client,
    session_factory: sessionmaker[Session],
    workspace_tmp_path: Path,
    monkeypatch,
):
    owner_id, owner_token = _create_user_and_token(client, "owner@example.com")
    _, other_token = _create_user_and_token(client, "other@example.com")

    monkeypatch.setattr(settings, "project_root", workspace_tmp_path)
    monkeypatch.setattr(settings, "upload_dir", Path("uploads"))
    gradcam_path = workspace_tmp_path / "uploads" / "gradcam" / "result.png"
    gradcam_path.parent.mkdir(parents=True)
    gradcam_path.write_bytes(b"private-gradcam")

    with session_factory() as database:
        prediction = Prediction(
            user_id=owner_id,
            model_id=None,
            predicted_class="class-a",
            confidence=0.95,
            top_predictions_json=[{"class": "class-a", "probability": 0.95}],
            gradcam_image_path="uploads/gradcam/result.png",
            low_confidence=False,
        )
        database.add(prediction)
        database.commit()
        prediction_id = prediction.id

    endpoint = f"/api/v1/predictions/{prediction_id}/gradcam"
    assert client.get(endpoint).status_code == 401
    assert client.get(endpoint, headers={"Authorization": f"Bearer {other_token}"}).status_code == 404

    response = client.get(endpoint, headers={"Authorization": f"Bearer {owner_token}"})
    assert response.status_code == 200
    assert response.content == b"private-gradcam"
    assert response.headers["cache-control"] == "private, no-store"
    assert response.headers["x-content-type-options"] == "nosniff"
