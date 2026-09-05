from app.main import app


def test_required_routes_are_registered():
    paths = set(app.openapi()["paths"])
    assert "/api/v1/models" in paths
    assert "/api/v1/experiments" in paths
    assert "/api/v1/analytics/overview" in paths
    assert "/media" not in paths


def test_upload_requires_authentication(client):
    response = client.post("/api/v1/predictions", files={"file": ("cell.png", b"not-an-image", "image/png")})
    assert response.status_code == 401


def test_research_routes_require_authentication(client):
    for path in (
        "/api/v1/models",
        "/api/v1/experiments",
        "/api/v1/predictions",
        "/api/v1/analytics/overview",
        "/api/v1/analytics/class-distribution",
        "/api/v1/analytics/model-performance",
    ):
        response = client.get(path)
        assert response.status_code == 401, path
