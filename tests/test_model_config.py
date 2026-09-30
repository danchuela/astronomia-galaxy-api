from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from openai import NOT_GIVEN

from packages.galaxy_agent import galaxy_agent
from packages.galaxy_agent.langchain_backend import LangChainBackend
from packages.galaxy_agent.models import AnalyzeRequest


@pytest.mark.parametrize("model", ["gpt-6-luna", "gpt-4.1"])
def test_parser_and_summary_request_compatibility(
    monkeypatch: pytest.MonkeyPatch, model: str
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_PARSE_MODEL", model)
    monkeypatch.setenv("OPENAI_MODEL", model)
    backend = LangChainBackend()
    client = Mock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"name":"M51","task":"cas","can_fulfill":true}')
            )
        ]
    )
    backend._client = client
    request = backend.enrich_request(AnalyzeRequest(request_id="test", message="CAS de M51"))
    assert request.target is not None and request.target.name == "M51"
    assert request.task == "cas"
    parse_kwargs = client.chat.completions.create.call_args.kwargs
    assert parse_kwargs["response_format"] == {"type": "json_object"}

    client.chat.completions.create.return_value.choices[0].message.content = "Resumen cualitativo."
    summary = backend.generate_accompanying_summary("M51", "visible", "CAS: 1.25")
    assert "1.25" in summary
    summary_kwargs = client.chat.completions.create.call_args.kwargs
    for kwargs in [parse_kwargs, summary_kwargs]:
        assert kwargs["model"] == model
        if model == "gpt-6-luna":
            assert kwargs["reasoning_effort"] == "none"
        else:
            assert kwargs["reasoning_effort"] is NOT_GIVEN


@pytest.mark.parametrize("model", [None, "gpt-4.1"])
def test_agent_tool_calling_configuration(
    monkeypatch: pytest.MonkeyPatch, model: str | None
) -> None:
    monkeypatch.delenv("AGENT_MODEL", raising=False)
    llm = Mock()
    create = Mock()
    monkeypatch.setattr(galaxy_agent, "ChatOpenAI", llm)
    monkeypatch.setattr(galaxy_agent, "create_agent", create)
    galaxy_agent.build_galaxy_agent([], model_name=model)
    assert llm.call_args.kwargs["model"] == (model or "gpt-6-luna")
    assert llm.call_args.kwargs["reasoning_effort"] == (None if model else "none")
    assert create.call_args.args[0] is llm.return_value


def test_parser_and_summary_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.delenv("OPENAI_PARSE_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    backend = LangChainBackend()
    assert backend._parse_model == backend._model == "gpt-6-luna"
