"""API tests through HTTP (uses FastAPI's TestClient): valid, hard, invalid, full flow."""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

CSV_PATH = Path(__file__).resolve().parent.parent / "data" / "registro_clases_asistencia.csv"


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


def post_predict(client, student, date):
    return client.post("/api/predict", json={"student": student, "date": date})


def test_valid_case(client):
    response = post_predict(client, "Marco Antonio", "15/05/2026")
    assert response.status_code == 200
    body = response.json()
    assert 0 <= body["probabilities"]["late_payment"] <= 1
    assert body["warnings"] == []


def test_hard_case(client):
    response = post_predict(client, "Lucas Garrón", "26/02/2026")
    assert response.status_code == 200
    assert response.json()["warnings"]


@pytest.mark.parametrize("student,date,status", [
    ("Pedro", "15/05/2026", 404),
    ("Marco Antonio", "hola", 422),
    ("", "15/05/2026", 422),
])
def test_invalid_input(client, student, date, status):
    response = post_predict(client, student, date)
    assert response.status_code == status
    assert response.json()["error"]


def test_malformed_body(client):
    response = client.post("/api/predict", content="no es json", headers={"Content-Type": "application/json"})
    assert response.status_code == 422


def test_full_flow_upload_train_predict(client):
    """Upload CSV -> train -> list students -> predict."""
    train = client.post("/api/train", content=CSV_PATH.read_bytes(), headers={"Content-Type": "text/csv"})
    assert train.status_code == 200
    assert "Marco Antonio" in train.json()["students"]
    assert len(train.json()["metrics"]) == 2
    assert client.get("/api/status").status_code == 200
    assert post_predict(client, "Marco Antonio", "2026-05-15").status_code == 200


def test_bad_csv_upload(client):
    response = client.post("/api/train", content=b"a,b\n1,2\n", headers={"Content-Type": "text/csv"})
    assert response.status_code == 400
    assert "Faltan columnas" in response.json()["error"]


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "Riesgo de mora" in response.text
