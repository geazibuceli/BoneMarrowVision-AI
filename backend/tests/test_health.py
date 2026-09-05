def test_health(client):
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["research_only"] is True


def test_disclaimer(client):
    response = client.get("/api/v1/disclaimer")
    assert "not a medical device" in response.json()["text"]
