"""English answers (phase 2 i18n): `answer_language` from the BFF selects the texts.

Spanish stays the default, so requests without the field behave as before.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from packages.galaxy_agent import messages
from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.agent_tools import make_tool_generate_final_report
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.langchain_backend import LangChainBackend
from packages.galaxy_agent.models import AnalyzeRequest, Target
from packages.galaxy_agent.orchestrator import TaskOrchestrator
from packages.galaxy_agent.request_utils import append_catalog_and_field
from packages.galaxy_core.application.isophotes import format_isophotes_summary
from packages.galaxy_core.domain.errors import TargetNotFoundError
from packages.galaxy_core.infrastructure.mast_hst_client import format_hst_jwst_info
from packages.galaxy_core.infrastructure.simbad_client import format_object_info

# Must match astronomia-ui-backend app/analysis_actions.py (VIEWER_READY_MARKER_EN, _LABEL_RE).
BFF_MARKER_EN = "When you are ready, tell me what you want to analyse"


def _request(lang: str | None = "en", **kw: Any) -> AnalyzeRequest:
    return AnalyzeRequest(request_id="r-en", message="Show me M51", answer_language=lang, **kw)


# ── Request model ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("value", "expected"), [("en", "en"), (" EN ", "en"), ("es", "es"), ("fr", None), (None, None)]
)
def test_answer_language_is_normalised(value: Any, expected: str | None) -> None:
    assert _request(value).answer_language == expected


def test_answer_language_survives_request_rebuilds() -> None:
    resolved = _request().to_resolved_request(target=Target(name="M51"), task="resolve")
    assert resolved.answer_language == "en"
    assert messages.lang_of(resolved) == "en"
    assert messages.lang_of(_request(None)) == "es"


# ── Fixed texts ──────────────────────────────────────────────────────


def test_viewer_ready_message_matches_the_bff_buttons() -> None:
    text = messages.viewer_ready_message("M51", "en")
    assert text.startswith("Here is M51. Adjust")  # BFF _LABEL_RE["en"]
    assert BFF_MARKER_EN in text
    short = messages.viewer_ready_message("M51", "en", list_analyses=False)
    assert short.endswith(BFF_MARKER_EN + ".")
    # Spanish unchanged
    assert messages.viewer_ready_message("M51") == (
        "Aquí tienes M51. Ajusta el encuadre, el zoom y la banda en el visor como prefieras. "
        "Cuando estés listo, dime qué quieres analizar: morfología completa, segmentación e "
        "isofotas, fotometría básica, o perfil de brillo."
    )


def test_every_status_message_has_an_english_version() -> None:
    assert messages.TOOL_STATUS["en"].keys() == messages.TOOL_STATUS["es"].keys()


def test_english_texts() -> None:
    assert messages.target_required_message("en").startswith("Tell me which galaxy")
    assert messages.target_not_found_message("M123", "en").startswith(
        'I could not find an object named "M123".'
    )
    assert append_catalog_and_field("Done.", "SDSS", 10, "en") == (
        "Done. Catalogue used: SDSS. Field: 10.0 arcmin."
    )
    assert append_catalog_and_field("Hecho.", "SDSS", 10) == (
        "Hecho. Catálogo usado: SDSS. Campo: 10.0 arcmin."
    )


def test_catalogue_formatters_in_english() -> None:
    record = {"main_id": "M 51", "otype_txt": "Seyfert2", "morph_type": "SA(s)bc", "sp_type": "G"}
    assert format_object_info(record, "en") == (
        "M 51 — type: Seyfert2. morphology: SA(s)bc. spectral type: G."
    )
    assert "tipo: Seyfert2" in format_object_info(record)
    assert format_hst_jwst_info({"collection": "HST"}, "en") == "Observation available in HST."
    table = [{"sma": 2.0, "ellip": 0.1}, {"sma": 20.0, "ellip": 0.3}]
    assert format_isophotes_summary(table, "M51", lang="en").startswith(
        "Isophotes for M51: 2 fitted ellipses"
    )
    assert format_isophotes_summary([], "M51", lang="en").startswith("Isointensity contours")


# ── Runner and orchestrator ──────────────────────────────────────────


def _runner(monkeypatch: pytest.MonkeyPatch, tmp_path: Any, agent: Any) -> AgentRunner:
    monkeypatch.setattr("packages.galaxy_agent.orchestrator.build_galaxy_agent", lambda _: agent)
    runner = AgentRunner.__new__(AgentRunner)
    runner.langsmith_enabled = False
    runner.langchain_backend = SimpleNamespace(enrich_request=lambda req: req)
    runner.orchestrator = TaskOrchestrator(
        analyzer=None, artifact_store=ArtifactStore(str(tmp_path))
    )
    return runner


def test_target_not_found_in_english(monkeypatch, tmp_path) -> None:
    class Agent:
        def stream(self, *a: Any, **k: Any) -> Any:
            raise TargetNotFoundError("M123")
            yield  # pragma: no cover

    runner = _runner(monkeypatch, tmp_path, Agent())
    request = _request(target=Target(name="M123"), task="resolve")
    events = list(runner.run_stream(request))
    assert events[-1]["summary"] == messages.target_not_found_message("M123", "en")


def test_status_and_errors_in_english(monkeypatch, tmp_path) -> None:
    class Agent:
        def stream(self, *a: Any, **k: Any) -> Any:
            call = SimpleNamespace(tool_calls=[{"name": "resolve_and_fetch_image"}])
            yield {"model": {"messages": [call]}}

    runner = _runner(monkeypatch, tmp_path, Agent())
    events = list(runner.run_stream(_request(target=Target(name="M51"), task="resolve")))
    assert events[0] == {"type": "status", "message": "Resolving target…"}
    assert events[-1] == {"type": "error", "message": messages.no_summary_message("en")}


def test_target_prompt_in_english() -> None:
    runner = AgentRunner.__new__(AgentRunner)
    runner.langsmith_enabled = False
    runner.langchain_backend = SimpleNamespace(enrich_request=lambda req: req)
    response = runner.run(_request(target=Target(name=""), task="resolve"))
    assert response.summary == messages.target_required_message("en")


def test_resolve_report_in_english() -> None:
    request = _request(target=Target(name="M51"), task="resolve")
    registry = ContextRegistry(request.request_id)
    registry.put(f"coordinates:{request.request_id}", {"ra_deg": 202.47, "dec_deg": 47.2})
    report = make_tool_generate_final_report(registry, None, request)
    assert report.invoke({})["summary"] == messages.viewer_ready_message("M51", "en")


# ── LLM parser and summary ───────────────────────────────────────────


class _FakeCompletions:
    def __init__(self, answer: dict[str, Any]) -> None:
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        message = SimpleNamespace(content=json.dumps(self.answer))
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _backend(answer: dict[str, Any]) -> tuple[LangChainBackend, _FakeCompletions]:
    backend = LangChainBackend.__new__(LangChainBackend)
    backend._parse_model = backend._model = "gpt-4o-mini"
    completions = _FakeCompletions(answer)
    backend._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return backend, completions


def test_parser_prompt_asks_for_english_decline_texts() -> None:
    backend, _ = _backend({})
    prompt = backend._build_system_prompt(lang="en")
    assert "decline_reason breve en inglés" in prompt
    assert "Use the interactive viewer buttons" in prompt
    assert "en español" not in prompt
    assert "en español" in backend._build_system_prompt()


@pytest.mark.parametrize(
    "answer",
    [
        {"can_fulfill": True, "name": "M51", "task": "segment"},
        {"can_fulfill": False, "decline_reason": "I cannot do spectra."},
    ],
)
def test_parser_keeps_the_answer_language(monkeypatch, answer: dict[str, Any]) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend, completions = _backend(answer)
    enriched = backend.enrich_request(_request())
    assert enriched.answer_language == "en"
    assert "inglés" in completions.calls[0]["messages"][0]["content"]


def test_english_summary_keeps_the_metrics_in_english(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    backend, _ = _backend({})
    metrics = "CAS — Concentration: 3.100, Asymmetry: 0.100, Smoothness: 0.050."
    assert backend.generate_accompanying_summary("M51", None, metrics, lang="en") == metrics
    spanish = backend.generate_accompanying_summary("M51", None, metrics)
    assert "Concentración (C)" in spanish
