"""Unknown targets are clarification responses, not module outages."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import requests
from fastapi.testclient import TestClient

from apps.api.main import app, get_runner
from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.agent_tools import _resolve_target
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.messages import target_not_found_message
from packages.galaxy_agent.models import AnalyzeRequest, Target
from packages.galaxy_agent.orchestrator import TaskOrchestrator
from packages.galaxy_core.domain.errors import (
    ImageIntegrityError,
    NoSourceDetectedError,
    TargetNotFoundError,
)
from packages.galaxy_core.infrastructure import sesame_client


@pytest.fixture
def request_body():
    return AnalyzeRequest(
        request_id="unknown-target",
        message="quiero analizar M123",
        target=Target(name="M123"),
        task="resolve",
    )


def _response(monkeypatch, text, status=200):
    def raise_for_status():
        if status >= 400:
            raise requests.HTTPError(f"HTTP {status}")

    monkeypatch.setattr(
        sesame_client.requests,
        "get",
        lambda *a, **kw: SimpleNamespace(text=text, raise_for_status=raise_for_status),
    )


def test_resolver_preserves_valid_coordinates(monkeypatch):
    _response(monkeypatch, "#=Simbad\n%J 148.888 69.065\n")
    assert sesame_client.resolve("M81") == (148.888, 69.065)


def test_unknown_name_is_typed(monkeypatch):
    _response(monkeypatch, "#!Sesame\n# NGC99999999\n#=Simbad: Nothing found\n")
    with pytest.raises(TargetNotFoundError) as caught:
        sesame_client.resolve(" NGC99999999 ")
    assert caught.value.name == "NGC99999999"


@pytest.mark.parametrize("name", ["M123", "M 123", "messier 123", "M0", "Messier 111"])
def test_invalid_messier_name_does_not_call_external_services(monkeypatch, name):
    def unexpected(*args, **kwargs):
        pytest.fail("Invalid Messier designations must not call the resolver")

    monkeypatch.setattr(sesame_client.requests, "get", unexpected)
    with pytest.raises(TargetNotFoundError):
        sesame_client.resolve(name)


@pytest.mark.parametrize("name", ["M1", "M110", "Messier 110", "m 81"])
def test_valid_messier_boundaries_are_resolved(monkeypatch, name):
    _response(monkeypatch, "%J 148.888 69.065\n")
    assert sesame_client.resolve(name) == (148.888, 69.065)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "<html>Temporarily unavailable</html>",
        "Unexpected upstream response",
        "# -ox\t#Q123\n#! *** NNNothing found ***\n\n# M81\t#Q123\nResolver failure\n",
    ],
)
def test_invalid_resolver_response_is_not_unknown_target(monkeypatch, text):
    _response(monkeypatch, text)
    with pytest.raises(RuntimeError, match="no usable position"):
        sesame_client.resolve("M81")


def test_real_sesame_not_found_payload(monkeypatch):
    name = "astronomia-no-such-target-123456789"
    _response(
        monkeypatch,
        "# -ox\t#Q123\n#! *** NNNothing found ***\n\n"
        f"# {name}\t#Q123\n#! *** NNNothing found ***\n\n#****Total of 1 crashes\n",
    )
    with pytest.raises(TargetNotFoundError):
        sesame_client.resolve(name)


def test_http_failure_is_not_unknown_target(monkeypatch):
    _response(monkeypatch, "Service unavailable", status=503)
    with pytest.raises(requests.HTTPError):
        sesame_client.resolve("M81")


def test_resolve_pipeline_keeps_typed_outcome(monkeypatch, request_body):
    def missing(**kwargs):
        raise TargetNotFoundError("M123")

    monkeypatch.setattr("packages.galaxy_agent.agent_tools.resolve_and_fetch", missing)
    with pytest.raises(TargetNotFoundError):
        _resolve_target(request_body, {})


def _runner(monkeypatch, tmp_path, failure):
    class Agent:
        def stream(self, *args, **kwargs):
            raise failure
            yield  # pragma: no cover

    monkeypatch.setattr("packages.galaxy_agent.orchestrator.build_galaxy_agent", lambda _: Agent())
    runner = AgentRunner.__new__(AgentRunner)
    runner.langsmith_enabled = False
    runner.langchain_backend = SimpleNamespace(enrich_request=lambda req: req)
    runner.orchestrator = TaskOrchestrator(
        analyzer=None,
        artifact_store=ArtifactStore(str(tmp_path)),
    )
    return runner


@pytest.mark.parametrize("task", ["resolve", "morphology_summary"])
def test_runner_returns_clarification_in_json(monkeypatch, tmp_path, request_body, task):
    runner = _runner(monkeypatch, tmp_path, TargetNotFoundError("M123"))
    response = runner.run(request_body.model_copy(update={"task": task}))
    assert response.status == "success"
    assert response.summary == target_not_found_message("M123")
    assert response.results["target_resolution"]["status"] == "not_found"
    assert response.artifacts == []


def test_runner_returns_complete_stream_without_coordinates(monkeypatch, tmp_path, request_body):
    runner = _runner(monkeypatch, tmp_path, TargetNotFoundError("M123"))
    events = list(runner.run_stream(request_body))
    assert [event["type"] for event in events] == ["summary", "end"]
    assert events[-1]["summary"] == events[0]["summary"] == target_not_found_message("M123")
    assert events[-1]["status"] == "success"
    assert events[-1]["artifacts"] == []
    assert "coordinates" not in events[-1]


def test_real_failure_is_still_an_error(monkeypatch, tmp_path, request_body):
    runner = _runner(monkeypatch, tmp_path, requests.Timeout("Resolver unavailable"))
    events = list(runner.run_stream(request_body))
    assert any(event["type"] == "error" for event in events)
    assert not any(event.get("status") == "success" for event in events)


@pytest.mark.parametrize(
    "failure,reason",
    [
        (NoSourceDetectedError("empty field"), "no_detection"),
        (ImageIntegrityError("stale capture"), "capture_required"),
    ],
)
def test_measurement_input_failure_is_a_clarification_not_an_outage(
    monkeypatch, tmp_path, request_body, failure, reason
):
    runner = _runner(monkeypatch, tmp_path, failure)
    req = request_body.model_copy(update={"task": "cas", "view_ra_deg": 12, "view_dec_deg": 20})
    events = list(runner.run_stream(req))
    assert [event["type"] for event in events] == ["summary", "end"]
    assert events[-1]["status"] == "success"
    assert events[-1]["analysis_quality"] == {"status": "unavailable", "reason": reason}
    assert events[-1]["artifacts"] == []


@pytest.mark.parametrize("route", ["/analyze", "/analyze/stream"])
def test_http_clarification_contract(monkeypatch, tmp_path, request_body, route):
    from apps.api.auth import verify_api_key

    runner = _runner(monkeypatch, tmp_path, TargetNotFoundError("M123"))
    app.dependency_overrides[get_runner] = lambda: runner
    app.dependency_overrides[verify_api_key] = lambda: None
    try:
        with TestClient(app) as client:
            response = client.post(route, json=request_body.model_dump())
        assert response.status_code == 200
        if route == "/analyze":
            payload = response.json()
        else:
            events = [
                json.loads(line[6:])
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]
            payload = events[-1]
        assert payload["status"] == "success"
        assert payload["summary"] == target_not_found_message("M123")
    finally:
        app.dependency_overrides.pop(get_runner, None)
        app.dependency_overrides.pop(verify_api_key, None)
