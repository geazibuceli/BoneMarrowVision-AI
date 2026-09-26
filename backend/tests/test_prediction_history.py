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


def test_history_filters_and_csv_are_consistent_and_owner_scoped(client, session_factory):
    import csv
    from io import StringIO

    from app.models import MLModel

    owner, token = _create_user_and_token(client, "filter@example.com")
    other, other_token = _create_user_and_token(client, "other-filter@example.com")
    with session_factory() as db:
        model = MLModel(name="=unsafe()", version="v1", architecture="baseline_cnn",
                        checkpoint_path="models/test.pt", config_path="ml/configs/test.yaml")
        db.add(model)
        db.flush()
        model_id = model.id
        for user_id, label, low, day in ((owner, "A", True, 2), (owner, "B", False, 2),
                                        (owner, "A", True, 3), (other, "A", True, 2)):
            db.add(Prediction(user_id=user_id, model_id=model.id, predicted_class=label, confidence=0.4,
                              low_confidence=low, top_predictions_json=[],
                              created_at=datetime(2026, 1, day, tzinfo=UTC)))
        db.commit()
    query = f"?predicted_class=A&low_confidence=true&model_id={model_id}&date_from=2026-01-02&date_to=2026-01-02"
    headers = {"Authorization": f"Bearer {token}"}
    result = client.get("/api/v1/predictions" + query, headers=headers)
    assert result.status_code == 200
    assert len(result.json()) == 1
    exported = client.get("/api/v1/predictions/export" + query, headers=headers)
    assert exported.status_code == 200
    rows = list(csv.DictReader(StringIO(exported.text)))
    assert [int(row["id"]) for row in rows] == [row["id"] for row in result.json()]
    assert rows[0]["model"] == "'=unsafe()"
    other_rows = client.get("/api/v1/predictions/export" + query,
                            headers={"Authorization": f"Bearer {other_token}"})
    assert rows[0]["id"] not in [row["id"] for row in csv.DictReader(StringIO(other_rows.text))]
    assert client.get("/api/v1/predictions/export").status_code == 401
    assert client.get("/api/v1/predictions?date_from=2026-02-01&date_to=2026-01-01", headers=headers).status_code == 422
