from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_reports_process_and_model_state() -> None:
    response = client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_ready": False}
