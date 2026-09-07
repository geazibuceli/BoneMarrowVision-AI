from app.core.config import settings
from app.core.rate_limit import rate_limiter


def test_auth_rate_limit_returns_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "auth_rate_limit", 1)
    monkeypatch.setattr(settings, "auth_rate_window_seconds", 60)
    rate_limiter.clear()
    try:
        first = client.post(
            "/api/v1/auth/login",
            data={"username": "unknown@example.com", "password": "wrong-password"},
        )
        second = client.post(
            "/api/v1/auth/login",
            data={"username": "unknown@example.com", "password": "wrong-password"},
        )
        assert first.status_code == 401
        assert second.status_code == 429
        assert int(second.headers["retry-after"]) >= 1
    finally:
        rate_limiter.clear()


def test_rate_limit_uses_trusted_proxy_ip(monkeypatch):
    from starlette.requests import Request

    monkeypatch.setattr(settings, "trust_proxy_headers", True)
    first_scope = {"type": "http", "client": ("proxy", 1), "headers": [(b"x-real-ip", b"10.0.0.1")]}
    second_scope = {"type": "http", "client": ("proxy", 1), "headers": [(b"x-real-ip", b"10.0.0.2")]}
    from app.core.rate_limit import _client_key

    assert _client_key(Request(first_scope), "auth") != _client_key(Request(second_scope), "auth")