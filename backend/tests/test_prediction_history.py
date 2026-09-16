from datetime import UTC, datetime

from test_media_security import _create_user_and_token

from app.models import Prediction


def test_history_paginates_tied_timestamps_and_isolates_owners(client, session_factory):
    owner_id, owner_token = _create_user_and_token(client, "history-owner@example.com")
    _, other_token = _create_user_and_token(client, "history-other@example.com")
    created_at = datetime(2026, 1, 1, tzinfo=UTC)
    with session_factory() as database:
        records = [
            Prediction(
                user_id=owner_id,
                predicted_class=f"class-{index}",
                confidence=0.9,
                top_predictions_json=[],
                created_at=created_at,
            )
            for index in range(105)
        ]
        database.add_all(records)
        database.commit()
        expected_ids = sorted((record.id for record in records), reverse=True)

    headers = {"Authorization": f"Bearer {owner_token}"}
    actual_ids = []
    for offset in range(0, 125, 25):
        response = client.get(f"/api/v1/predictions?limit=26&offset={offset}", headers=headers)
        assert response.status_code == 200
        actual_ids.extend(record["id"] for record in response.json()[:25])
    assert actual_ids == expected_ids
    assert client.get("/api/v1/predictions?offset=105", headers=headers).json() == []

    detail_path = f"/api/v1/predictions/{expected_ids[0]}"
    assert client.get(detail_path).status_code == 401
    other_headers = {"Authorization": f"Bearer {other_token}"}
    assert client.get(detail_path, headers=other_headers).status_code == 404
    assert client.get("/api/v1/predictions", headers=other_headers).json() == []
    assert client.get("/api/v1/predictions?offset=-1", headers=headers).status_code == 422
