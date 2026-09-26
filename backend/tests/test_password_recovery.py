import hashlib
from datetime import UTC, datetime, timedelta

from test_media_security import _create_user_and_token

from app.core.config import settings
from app.models import User


def test_recovery_is_private_single_use_and_revokes_sessions(client, session_factory, monkeypatch):
    user_id, token = _create_user_and_token(client, "recovery@example.com")
    sent = []
    monkeypatch.setattr(settings, "smtp_host", "smtp.example.com")
    monkeypatch.setattr(settings, "smtp_from", "app@example.com")
    monkeypatch.setattr("app.routers.auth.send_recovery_email", lambda email, value: sent.append((email, value)))
    known = client.post("/api/v1/auth/forgot-password", json={"email": "recovery@example.com"})
    unknown = client.post("/api/v1/auth/forgot-password", json={"email": "missing@example.com"})
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()
    assert len(sent) == 1
    reset_token = sent[0][1]
    with session_factory() as db:
        user = db.get(User, user_id)
        assert user.reset_token_hash == hashlib.sha256(reset_token.encode()).hexdigest()
        assert reset_token not in known.text
    payload = {"token": reset_token, "password": "Updated-Password-123!"}
    assert client.post("/api/v1/auth/reset-password", json=payload).status_code == 200
    assert client.post("/api/v1/auth/reset-password", json=payload).status_code == 400
    assert client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"}).status_code == 401
    login = client.post("/api/v1/auth/login", data={
        "username": "recovery@example.com", "password": payload["password"],
    })
    assert login.status_code == 200
    headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200


def test_expired_token_and_wrong_current_password_are_rejected(client, session_factory):
    user_id, token = _create_user_and_token(client, "expired-reset@example.com")
    value = "a" * 43
    with session_factory() as db:
        user = db.get(User, user_id)
        user.reset_token_hash = hashlib.sha256(value.encode()).hexdigest()
        user.reset_token_expires_at = datetime.now(UTC) - timedelta(seconds=1)
        db.commit()
    response = client.post("/api/v1/auth/reset-password", json={"token": value, "password": "New-password123"})
    assert response.status_code == 400
    headers = {"Authorization": f"Bearer {token}"}
    assert client.post("/api/v1/auth/change-password", headers=headers, json={
        "current_password": "wrong-password", "password": "New-password123",
    }).status_code == 400
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 200


def test_change_password_revokes_all_sessions_and_pending_reset(client, session_factory):
    user_id, token = _create_user_and_token(client, "change@example.com")
    from app.security import hash_password
    with session_factory() as db:
        user = db.get(User, user_id)
        user.password_hash = hash_password("Old-password123")
        user.reset_token_hash = "a" * 64
        db.commit()
    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/api/v1/auth/change-password", headers=headers, json={
        "current_password": "Old-password123", "password": "New-password123",
    })
    assert response.status_code == 200
    assert client.get("/api/v1/auth/me", headers=headers).status_code == 401
    with session_factory() as db:
        assert db.get(User, user_id).reset_token_hash is None


def test_unconfigured_recovery_is_explicit(client, monkeypatch):
    monkeypatch.setattr(settings, "smtp_host", "")
    assert client.post("/api/v1/auth/forgot-password", json={"email": "unknown@example.com"}).status_code == 503
