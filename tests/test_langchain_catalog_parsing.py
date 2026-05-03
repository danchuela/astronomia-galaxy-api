from __future__ import annotations

import json

import pytest

from packages.galaxy_agent.domain.models import AnalyzeRequest
from packages.galaxy_agent.langchain_backend import (
    LangChainBackend,
    _extract_catalog_from_text,
)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]


class _FakeCompletions:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs: object) -> _FakeResponse:
        return _FakeResponse(self._content)


class _FakeChat:
    def __init__(self, content: str) -> None:
        self.completions = _FakeCompletions(content)


class _FakeClient:
    def __init__(self, content: str) -> None:
        self.chat = _FakeChat(content)


def test_extract_catalog_from_text_supports_aliases() -> None:
    assert _extract_catalog_from_text("M104 con catálogo DSS2") == "DSS2"
    assert _extract_catalog_from_text("m104 DSS2-BLUE") == "DSS2-BLUE"
    assert _extract_catalog_from_text("M104 DSS2 azul") == "DSS2-BLUE"
    assert _extract_catalog_from_text("M104 DSS2 IR") == "DSS2-IR"
    assert _extract_catalog_from_text("M104 con 2MASS J") == "2MASS-J"
    assert _extract_catalog_from_text("M104 en GALEX") == "GALEX"


def test_enrich_request_uses_catalog_from_user_text(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = LangChainBackend()

    llm_payload = json.dumps(
        {
            "can_fulfill": True,
            "decline_reason": None,
            "name": "M104",
            "ra_deg": None,
            "dec_deg": None,
            "catalog": None,
            "band": "visible",
            "size_arcmin": 10.0,
            "task": None,
        }
    )
    backend._client = _FakeClient(llm_payload)  # type: ignore[assignment]

    request = AnalyzeRequest(
        request_id="req-1",
        message="Muéstrame m104con el catálogo DSS2-BLUE",
    )
    enriched = backend.enrich_request(request)

    assert enriched.options.get("catalog") == "DSS2-BLUE"
    assert "band" not in enriched.options
