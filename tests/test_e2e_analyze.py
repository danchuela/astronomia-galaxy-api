"""HTTP contracts with an isolated runner; no external services or API credits."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.config import get_settings
from apps.api.main import app, get_runner
from packages.galaxy_agent.models import AnalyzeResponse, Provenance


class _Runner:
    def run(self, request):
        return AnalyzeResponse(
            request_id=request.request_id,
            status="success",
            summary="Contract response",
            provenance=Provenance(versions={}),
        )

    def run_stream(self, request):
        yield {"type": "status", "message": "Processing"}
        yield {"type": "end", **self.run(request).model_dump()}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[TestClient]:
    monkeypatch.setenv("API_KEY", "contract-test-key")
    monkeypatch.setenv("REQUIRE_API_KEY", "true")
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    get_settings.cache_clear()
    app.dependency_overrides[get_runner] = _Runner
    try:
        with TestClient(app, headers={"X-API-Key": "contract-test-key"}) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.pop(get_runner, None)
        get_settings.cache_clear()


E2E_BODY = {
    "request_id": "test-1",
    "message": "Dame una imagen de M87 en visible",
}


def test_analyze_contract_nl_prompt(client: TestClient) -> None:
    """POST /analyze con mensaje NL; respuesta 200 y forma esperada."""
    r = client.post("/analyze", json=E2E_BODY)
    assert r.status_code == 200, r.text

    data = r.json()
    assert data["request_id"] == "test-1"
    assert data["status"] in ("success", "error")
    assert "summary" in data
    assert "artifacts" in data
    assert "provenance" in data

    if data["status"] == "success":
        assert isinstance(data["artifacts"], list)


def test_health(client: TestClient) -> None:
    """GET /health responde ok."""
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_analyze_rejects_invalid_key(client: TestClient) -> None:
    response = client.post("/analyze", json=E2E_BODY, headers={"X-API-Key": "wrong"})
    assert response.status_code == 401


def test_analyze_rejects_invalid_body(client: TestClient) -> None:
    assert client.post("/analyze", json={"request_id": "invalid"}).status_code == 422


def test_analyze_stream_contract(client: TestClient) -> None:
    response = client.post("/analyze/stream", json=E2E_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = [
        json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["status", "end"]
    assert events[-1]["status"] == "success"


@pytest.mark.parametrize(
    ("filename", "route", "media_type", "payload"),
    [
        ("image.jpg", "image", "image/jpeg", b"jpeg-contract-content"),
        ("plot-cas.png", "plot/cas", "image/png", b"png-contract-content"),
    ],
)
def test_artifact_contract(client, tmp_path, filename, route, media_type, payload) -> None:
    folder = tmp_path / "contract-artifact"
    folder.mkdir()
    (folder / filename).write_bytes(payload)
    response = client.get(f"/artifacts/contract-artifact/{route}")
    assert response.status_code == 200
    assert response.headers["content-type"] == media_type
    assert response.content == payload
