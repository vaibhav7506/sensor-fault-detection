from fastapi.testclient import TestClient
from sensor.platform import train_csv
def test_api_health_prediction_and_invalid_features(dataset, tmp_path, monkeypatch):
    _, path = dataset; train_csv(path, tmp_path); monkeypatch.setenv("MODEL_PATH", str(tmp_path / "model.joblib"))
    from app.main import app
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "healthy", "model_loaded": True}
        home = client.get("/")
        assert home.status_code == 200
        assert "Sensor Fault Detection API" in home.text
        assert client.get("/ready").json() == {"status": "ready"}
        favicon = client.get("/favicon.svg")
        assert favicon.status_code == 200
        assert favicon.headers["content-type"] == "image/svg+xml"
        features = {f: 0.0 for f in client.get("/model/info").json()["features"]}
        assert client.post("/predict", json={"features": features}).status_code == 200
        assert client.post("/predict", json={"features": {}}).status_code == 422
