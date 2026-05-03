"""Contrato de /analyze (TestClient, sin servidor real).

Para E2E real (API en marcha + imagen en disco) usa:
  python scripts/e2e_real.py
con la API levantada (docker compose up o make run).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from apps.api.main import app

client = TestClient(app)

E2E_BODY = {
    "request_id": "test-1",
    "message": "Dame una imagen de M87 en visible",
}


def test_analyze_contract_nl_prompt() -> None:
    """POST /analyze con mensaje NL; respuesta 200 y forma esperada."""
    r = client.post("/analyze", json=E2E_BODY, timeout=120)
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["request_id"] == "test-1"
    assert data["status"] in ("success", "error")
    assert "summary" in data
    assert "artifacts" in data
    assert "provenance" in data

    if data["status"] == "success":
        assert isinstance(data["artifacts"], list)


def test_health() -> None:
    """GET /health responde ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
