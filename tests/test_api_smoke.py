from fastapi.testclient import TestClient
from src.api.app import app

client = TestClient(app)

def test_health_ok():
    res = client.get("/health")
    assert res.status_code == 200
    body = res.json()
    assert "status" in body
    assert body["status"] == "ok"
    assert "model_name" in body