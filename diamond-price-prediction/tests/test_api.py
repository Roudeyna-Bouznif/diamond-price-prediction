import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

PAYLOAD = {"carat": 1.0, "cut": "Ideal", "color": "G", "clarity": "VS2",
           "depth": 61.8, "table": 57.0, "x": 6.4, "y": 6.4, "z": 3.96}


@pytest.fixture()
def client(trained, monkeypatch):
    out, _ = trained
    import diamonds.api as api
    monkeypatch.setattr(api, "MODEL_PATH", str(out / "models" / "model.joblib"))
    with TestClient(api.app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict(client):
    r = client.post("/predict", json=PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["lower_usd"] <= body["price_usd"] <= body["upper_usd"]
    assert body["confidence_level"] == pytest.approx(0.9)


def test_batch(client):
    r = client.post("/predict/batch", json=[PAYLOAD, {**PAYLOAD, "carat": 2.0, "x": 8.1,
                                                      "y": 8.1, "z": 5.0}])
    assert r.status_code == 200 and len(r.json()) == 2


def test_validation_error(client):
    assert client.post("/predict", json={**PAYLOAD, "carat": -1}).status_code == 422
    assert client.post("/predict", json={**PAYLOAD, "cut": "Shiny"}).status_code == 422


def test_model_info(client):
    assert "sklearn_version" in client.get("/model-info").json()
