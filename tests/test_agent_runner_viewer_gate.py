from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.models import AnalyzeRequest, AnalyzeResponse, Target
from packages.galaxy_agent.provenance_utils import build_provenance


class _StubBackend:
    def __init__(self, enriched: AnalyzeRequest) -> None:
        self.enriched = enriched

    def enrich_request(self, request: AnalyzeRequest) -> AnalyzeRequest:
        return self.enriched


class _StubOrchestrator:
    def __init__(self) -> None:
        self.requests: list[AnalyzeRequest] = []

    def run(self, request: AnalyzeRequest, langsmith_enabled: bool) -> AnalyzeResponse:
        self.requests.append(request)
        return AnalyzeResponse(
            request_id=request.request_id,
            status="success",
            summary=f"task={request.task}",
            results={},
            artifacts=[],
            provenance=build_provenance(langsmith_enabled),
            warnings=[],
        )

    def run_stream(
        self, request: AnalyzeRequest, langsmith_enabled: bool
    ) -> Iterator[dict[str, Any]]:
        self.requests.append(request)
        yield {
            "type": "end",
            "request_id": request.request_id,
            "status": "success",
            "summary": f"task={request.task}",
            "results": {},
            "artifacts": [],
            "provenance": build_provenance(langsmith_enabled).model_dump(),
            "warnings": [],
        }


def _runner(enriched: AnalyzeRequest) -> tuple[AgentRunner, _StubOrchestrator]:
    runner = AgentRunner.__new__(AgentRunner)
    runner.langsmith_enabled = False
    runner.langchain_backend = _StubBackend(enriched)
    orchestrator = _StubOrchestrator()
    runner.orchestrator = orchestrator
    return runner, orchestrator


def test_analysis_without_viewer_is_forced_to_resolve() -> None:
    enriched = AnalyzeRequest(
        request_id="req-1",
        message="quiero analizar M81",
        target=Target(name="M81"),
        task="morphology_summary",
    )
    runner, orchestrator = _runner(enriched)

    response = runner.run(enriched)

    assert response.summary == "task=resolve"
    assert orchestrator.requests[0].task == "resolve"
    assert orchestrator.requests[0].target == Target(name="M81")


def test_analysis_with_viewer_can_continue() -> None:
    enriched = AnalyzeRequest(
        request_id="req-2",
        message="analiza M81",
        target=Target(name="M81"),
        task="morphology_summary",
        view_ra_deg=148.888,
        view_dec_deg=69.065,
        view_size_arcmin=10.0,
        view_hips_id="CDS/P/DSS2/color",
    )
    runner, orchestrator = _runner(enriched)

    response = runner.run(enriched)

    assert response.summary == "task=morphology_summary"
    assert orchestrator.requests[0].task == "morphology_summary"


def test_analysis_with_viewer_uses_current_framed_field() -> None:
    enriched = AnalyzeRequest(
        request_id="req-3",
        message="analiza esta zona",
        target=Target(name="M81"),
        task="morphology_summary",
        view_ra_deg=187.706,
        view_dec_deg=12.391,
        view_size_arcmin=10.0,
        view_hips_id="CDS/P/DSS2/color",
    )
    runner, orchestrator = _runner(enriched)

    response = runner.run(enriched)

    assert response.summary == "task=morphology_summary"
    assert orchestrator.requests[0].task == "morphology_summary"
