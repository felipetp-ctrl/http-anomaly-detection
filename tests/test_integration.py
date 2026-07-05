"""Integration test: predict endpoint logs features correctly."""
from fastapi.testclient import TestClient


def test_predict_logs_features(tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    import monitoring.logger as logger_mod

    original_init = logger_mod.FeatureLogger.__init__

    def patched_init(self, connection_string=None, container_name="feature-logs", local_dir=None):
        original_init(self, connection_string=None, container_name=container_name, local_dir=tmp_path)

    monkeypatch.setattr(logger_mod.FeatureLogger, "__init__", patched_init)

    from api.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json={
            "ip": "10.0.0.1",
            "timestamp": 1751500000.0,
            "endpoint": "/api/test",
            "method": "GET",
            "status_code": 200,
            "payload_size": 1000,
            "user_agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "response_time": 50,
        })

        assert response.status_code == 200
        data = response.json()
        assert "is_anomaly" in data
        assert "anomaly_score" in data

    csv_files = list(tmp_path.glob("*.csv"))
    assert len(csv_files) >= 1


def test_predict_works_without_logger(monkeypatch):
    """Predict should work even if logger is not configured."""
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    from api.main import app

    with TestClient(app) as client:
        response = client.post("/predict", json={
            "ip": "10.0.0.2",
            "timestamp": 1751500000.0,
            "endpoint": "/api/test",
            "method": "GET",
            "status_code": 200,
            "payload_size": 1000,
            "user_agent": "Mozilla/5.0",
            "response_time": 50,
        })

        assert response.status_code == 200
