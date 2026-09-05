import pytest
from pydantic import SecretStr
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.core.config import settings
from app.models import AuditLog, MLModel, User


def _register(client, email: str, *, bootstrap_token: str | None = None):
    payload = {
        "email": email,
        "password": "correct-horse-battery-staple",
        "full_name": "Research User",
    }
    if bootstrap_token is not None:
        payload["admin_bootstrap_token"] = bootstrap_token
    return client.post("/api/v1/auth/register", json=payload)


def _login(client, email: str, password: str = "correct-horse-battery-staple") -> str:
    response = client.post(
        "/api/v1/auth/login",
        data={"username": email, "password": password},
    )
    assert response.status_code == 200
    return response.json()["access_token"]


def test_register_login_and_me_use_argon2(client, session_factory):
    response = _register(client, "USER@Example.com")
    assert response.status_code == 201
    assert response.json()["email"] == "user@example.com"
    assert response.json()["role"] == "researcher"

    with session_factory() as database:
        user = database.scalar(select(User).where(User.email == "user@example.com"))
        assert user is not None
        assert user.password_hash.startswith("$argon2")
        assert "correct-horse-battery-staple" not in user.password_hash

    bad_login = client.post(
        "/api/v1/auth/login",
        data={"username": "user@example.com", "password": "wrong-password"},
    )
    assert bad_login.status_code == 401
    assert bad_login.headers["www-authenticate"] == "Bearer"

    token = _login(client, " USER@example.com ")
    me = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert me.status_code == 200
    assert me.json()["email"] == "user@example.com"

    with session_factory() as database:
        actions = list(database.scalars(select(AuditLog.action).order_by(AuditLog.id)))
        assert actions == ["user.registered", "auth.login_succeeded"]


def test_registration_rejects_duplicate_email_and_blank_normalized_name(client):
    assert _register(client, "duplicate@example.com").status_code == 201
    assert _register(client, "DUPLICATE@example.com").status_code == 409

    response = client.post(
        "/api/v1/auth/register",
        json={
            "email": "blank-name@example.com",
            "password": "correct-horse-battery-staple",
            "full_name": "  ",
        },
    )
    assert response.status_code == 422


def test_admin_registration_requires_one_time_bootstrap_token(client, session_factory, monkeypatch):
    bootstrap_token = "a-secure-bootstrap-token-used-only-for-tests"
    monkeypatch.setattr(settings, "admin_bootstrap_token", SecretStr(bootstrap_token))

    assert _register(client, "bad-admin@example.com", bootstrap_token="wrong-token").status_code == 403

    admin = _register(client, "admin@example.com", bootstrap_token=bootstrap_token)
    assert admin.status_code == 201
    assert admin.json()["role"] == "admin"

    second_admin = _register(client, "second-admin@example.com", bootstrap_token=bootstrap_token)
    assert second_admin.status_code == 409

    with session_factory() as database:
        admin_count = database.scalar(select(func.count(User.id)).where(User.role == "admin"))
        assert admin_count == 1


def test_researcher_cannot_use_admin_model_registration(client):
    assert _register(client, "researcher@example.com").status_code == 201
    token = _login(client, "researcher@example.com")
    response = client.post(
        "/api/v1/models/register",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Model",
            "version": "1",
            "architecture": "baseline_cnn",
            "checkpoint_path": "models/model.pt",
            "config_path": "ml/configs/baseline.yaml",
        },
    )
    assert response.status_code == 403


def test_list_endpoints_enforce_pagination_bounds(client):
    assert _register(client, "pagination@example.com").status_code == 201
    token = _login(client, "pagination@example.com")
    headers = {"Authorization": f"Bearer {token}"}

    for path in ("/api/v1/models", "/api/v1/experiments", "/api/v1/predictions"):
        assert client.get(f"{path}?limit=1&offset=0", headers=headers).status_code == 200
        assert client.get(f"{path}?limit=501", headers=headers).status_code == 422


def test_database_enforces_only_one_active_model(session_factory):
    with session_factory() as database:
        database.add_all(
            [
                MLModel(
                    name=f"Model {index}",
                    version=str(index),
                    architecture="baseline_cnn",
                    checkpoint_path=f"models/model-{index}.pt",
                    config_path="ml/configs/baseline.yaml",
                    metrics_json={},
                    is_active=True,
                )
                for index in range(2)
            ]
        )
        with pytest.raises(IntegrityError):
            database.commit()
