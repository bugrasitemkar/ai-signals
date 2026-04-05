import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_endpoint_returns_200():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert "ollama" in data
    assert "huggingface" in data
    assert "status" in data["ollama"]
    assert "status" in data["huggingface"]
