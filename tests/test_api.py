import app as service_app


def test_index_returns_form(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "numberForm" in response.get_data(as_text=True)


def test_predict_returns_number(client, auth_headers, flat):
    response = client.post("/api/numbers", json=flat, headers=auth_headers)
    body = response.get_json()
    assert response.status_code == 200
    assert body["status"] == "success"
    assert isinstance(body["data"], int)
    assert body["data"] > 0


def test_predict_accepts_string_values(client, auth_headers):
    payload = {"area": "45.5", "rooms": "2", "total_floors": "16", "floor": "7"}
    response = client.post("/api/numbers", json=payload, headers=auth_headers)
    assert response.status_code == 200


def test_predict_requires_token(client, flat):
    assert client.post("/api/numbers", json=flat).status_code == 401


def test_predict_rejects_wrong_token(client, flat):
    headers = {"Authorization": "Bearer wrong"}
    assert client.post("/api/numbers", json=flat, headers=headers).status_code == 401


def test_predict_rejects_invalid_value(client, auth_headers, flat):
    response = client.post(
        "/api/numbers", json={**flat, "area": "abc"}, headers=auth_headers
    )
    assert response.status_code == 400
    assert response.get_json()["status"] == "error"


def test_predict_rejects_missing_field(client, auth_headers, flat):
    flat.pop("rooms")
    response = client.post("/api/numbers", json=flat, headers=auth_headers)
    assert response.status_code == 400


def test_untrained_model_returns_501(client, auth_headers, flat):
    def untrained(X):
        raise NotImplementedError("модель ещё не обучена")

    service_app.app.config["predict"] = untrained
    response = client.post("/api/numbers", json=flat, headers=auth_headers)
    assert response.status_code == 501


def test_metrics_endpoint(client, auth_headers, flat):
    client.post("/api/numbers", json=flat, headers=auth_headers)
    response = client.get("/metrics")
    text = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "flat_requests_total" in text
    assert "flat_request_duration_seconds_bucket" in text
    assert "flat_predicted_price_rub_count" in text
