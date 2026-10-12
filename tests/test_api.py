"""
Integration tests for FastAPI REST API endpoints using TestClient.
"""

import pytest
from fastapi.testclient import TestClient
from api import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_api_health_check(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "degraded"]
    assert "thresholds" in data
    assert "model_loaded" in data


def test_api_predict_valid_transaction(client):
    payload = {
        "Time": 450.0,
        "Amount": 125.50,
        "V1": -1.2,
        "V2": 0.8,
        "V3": -0.5,
        "V4": 1.1,
        "V14": -0.9
    }
    response = client.post("/predict?save_to_db=true", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "transaction_id" in data
    assert "fraud_probability" in data
    assert "risk_level" in data
    assert "prediction" in data
    assert "explanation" in data
    assert data["risk_level"] in ["LOW", "MEDIUM", "HIGH"]


def test_api_predict_negative_amount_validation_error(client):
    payload = {
        "Time": 100.0,
        "Amount": -50.0  # Invalid negative amount
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Pydantic validation error


def test_api_predict_batch(client):
    payload = {
        "transactions": [
            {"Time": 10.0, "Amount": 15.0},
            {"Time": 20.0, "Amount": 250.0, "V4": 3.8, "V14": -4.2}
        ]
    }
    response = client.post("/predict-batch?save_to_db=false", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_processed"] == 2
    assert len(data["predictions"]) == 2
    assert "fraud_rate_percentage" in data


def test_api_predict_batch_empty_fails(client):
    payload = {"transactions": []}
    response = client.post("/predict-batch", json=payload)
    assert response.status_code == 400


def test_api_get_transactions_history(client):
    response = client.get("/transactions?limit=10")
    assert response.status_code == 200
    data = response.json()
    assert "transactions" in data
    assert "count" in data


def test_api_get_metrics(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "live_database_kpis" in data
    assert "offline_model_metrics" in data
