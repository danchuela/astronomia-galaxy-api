from __future__ import annotations

import asyncio
import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi import FastAPI
from starlette.responses import StreamingResponse

from apps.api.capacity import AnalysisCapacity
from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.agent_tools import _resolve_target
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.messages import target_resolution_unavailable_message
from packages.galaxy_agent.models import AnalyzeRequest, Target
from packages.galaxy_agent.orchestrator import TaskOrchestrator
from packages.galaxy_core.domain.errors import TargetNotFoundError, TargetResolutionUnavailableError
from packages.galaxy_core.infrastructure import sesame_client as resolver

CLIENT = httpx.AsyncClient
MIRROR = """Object M 31 --- AGN
Coordinates(ICRS,ep=J2000,eq=2000): 00 42 44.330  +41 16 07.50 (IR) C
Coordinates(FK4,ep=B1950,eq=1950): 00 40 00.095 +40 59 41.73
Coordinates(Gal,ep=J2000,eq=2000): 121.174329 -21.573309
"""


def transport(monkeypatch, handler):
    monkeypatch.setattr(
        resolver.httpx,
        "AsyncClient",
        lambda **kw: CLIENT(transport=httpx.MockTransport(handler), **kw),
    )


@pytest.mark.parametrize("failure", [httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError])
def test_primary_failure_uses_independent_mirror(monkeypatch, failure):
    calls = []

    def handle(request):
        calls.append(request)
        if request.url.host == "cds.unistra.fr":
            raise failure("upstream down")
        return httpx.Response(200, text=MIRROR)

    transport(monkeypatch, handle)
    assert resolver.resolve("Andromeda") == pytest.approx((10.6847083333333, 41.26875))
    assert [call.url.host for call in calls] == ["cds.unistra.fr", "simbad.cfa.harvard.edu"]
    assert calls[0].url.path.endswith("/SNV")
    assert "-ox&" not in str(calls[0].url)


def test_primary_success_does_not_query_mirror(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, text="%J 148.888 69.065\n")

    transport(monkeypatch, handle)
    assert resolver.resolve("M81") == (148.888, 69.065)
    assert len(calls) == 1


def test_names_cannot_inject_extra_queries_or_options(monkeypatch):
    calls = []

    def handle(request):
        calls.append(request)
        return httpx.Response(200, text="%J 148.888 69.065\n")

    transport(monkeypatch, handle)
    resolver.resolve("M81&-ox/extra?")
    assert b"&" not in calls[0].url.query
    assert b"%26" in calls[0].url.query


@pytest.mark.parametrize(
    "position",
    ["%J 361 0", "%J 12 91", "%J -1 -90", "%J 9999999999999999999999999999999999999999999999 -91"],
)
def test_invalid_coordinates_are_never_returned(monkeypatch, position):
    def handle(request):
        return httpx.Response(
            200, text=position if request.url.host == "cds.unistra.fr" else "invalid"
        )

    transport(monkeypatch, handle)
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("M81")


def test_mirror_never_uses_fk4_or_galactic_coordinates(monkeypatch):
    transport(
        monkeypatch,
        lambda req: (
            httpx.Response(503)
            if req.url.host == "cds.unistra.fr"
            else httpx.Response(200, text=MIRROR.replace("ICRS", "OTHER"))
        ),
    )
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("M31")


def test_mirror_multiple_matches_are_not_silently_selected(monkeypatch):
    transport(
        monkeypatch,
        lambda req: (
            httpx.Response(503)
            if req.url.host == "cds.unistra.fr"
            else httpx.Response(200, text=MIRROR + MIRROR)
        ),
    )
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("M31")


def test_mirror_negative_answer_does_not_invent_global_not_found(monkeypatch):
    transport(
        monkeypatch,
        lambda req: (
            httpx.Response(503)
            if req.url.host == "cds.unistra.fr"
            else httpx.Response(200, text="Identifier not found")
        ),
    )
    with pytest.raises(TargetResolutionUnavailableError) as caught:
        resolver.resolve("unknown-source")
    assert not isinstance(caught.value, TargetNotFoundError)


def test_whole_lookup_deadline_cancels_the_upstream_work(monkeypatch):
    cancelled = threading.Event()

    async def handle(request):
        try:
            await asyncio.sleep(3600)
        finally:
            cancelled.set()

    transport(monkeypatch, handle)
    monkeypatch.setenv("SESAME_TIMEOUT", ".05")
    start = time.monotonic()
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("M31")
    assert time.monotonic() - start < 1
    assert cancelled.is_set()


def test_primary_cannot_consume_the_mirrors_deadline(monkeypatch):
    cancelled = threading.Event()

    async def handle(request):
        if request.url.host == "cds.unistra.fr":
            try:
                await asyncio.sleep(3600)
            finally:
                cancelled.set()
        return httpx.Response(200, text=MIRROR)

    transport(monkeypatch, handle)
    monkeypatch.setenv("SESAME_TIMEOUT", ".2")
    start = time.monotonic()
    assert resolver.resolve("M31") == pytest.approx((10.6847083333333, 41.26875))
    assert time.monotonic() - start < 1
    assert cancelled.is_set()


def test_large_responses_are_bounded(monkeypatch):
    transport(
        monkeypatch, lambda req: httpx.Response(200, text="x" * (resolver.MAX_RESPONSE_BYTES + 1))
    )
    with pytest.raises(TargetResolutionUnavailableError):
        resolver.resolve("M31")


def test_resolution_failure_is_not_retried_as_another_survey(monkeypatch):
    def fail(**kwargs):
        raise TargetResolutionUnavailableError("M31")

    monkeypatch.setattr("packages.galaxy_agent.agent_tools.resolve_and_fetch", fail)
    request = AnalyzeRequest(request_id="resolver-down", target=Target(name="M31"), task="resolve")
    with pytest.raises(TargetResolutionUnavailableError):
        _resolve_target(request, {})


@pytest.mark.parametrize("lang", ["es", "en"])
def test_resolver_failure_ends_json_and_stream_with_specific_error(monkeypatch, tmp_path, lang):
    class Agent:
        def stream(self, *args, **kwargs):
            raise TargetResolutionUnavailableError("M31")
            yield

    monkeypatch.setattr("packages.galaxy_agent.orchestrator.build_galaxy_agent", lambda _: Agent())
    runner = AgentRunner.__new__(AgentRunner)
    runner.langsmith_enabled = False
    runner.langchain_backend = SimpleNamespace(enrich_request=lambda request: request)
    runner.orchestrator = TaskOrchestrator(
        analyzer=None, artifact_store=ArtifactStore(str(tmp_path))
    )
    request = AnalyzeRequest(
        request_id="resolver-down", target=Target(name="M31"), task="resolve", answer_language=lang
    )
    events = list(runner.run_stream(request))
    assert [event["type"] for event in events] == ["summary", "end"]
    assert events[-1]["error_code"] == "target_resolution_unavailable"
    assert events[-1]["status"] == "error"
    assert events[-1]["summary"] == target_resolution_unavailable_message(lang)
    assert "coordinates" not in events[-1]
    response = runner.run(request)
    assert response.status == "error" and response.summary == events[-1]["summary"]


def test_expired_lookup_releases_capacity_before_next_request(monkeypatch):
    started = threading.Event()
    cancelled = threading.Event()
    requests = 0

    async def handle(request):
        nonlocal requests
        requests += 1
        if requests <= 2:
            started.set()
            try:
                await asyncio.sleep(3600)
            finally:
                cancelled.set()
        return httpx.Response(200, text="%J 148.888 69.065\n")

    transport(monkeypatch, handle)
    monkeypatch.setenv("SESAME_TIMEOUT", ".15")
    small = FastAPI()
    small.add_middleware(AnalysisCapacity)

    @small.get("/health")
    def health():
        return {"status": "ok"}

    @small.post("/analyze/stream")
    def analyze():
        def events():
            try:
                resolver.resolve("M31")
                status = "success"
            except TargetResolutionUnavailableError:
                status = "error"
            yield "data: " + json.dumps({"type": "end", "status": status}) + "\n\n"

        return StreamingResponse(events(), media_type="text/event-stream")

    async def check():
        async with CLIENT(
            transport=httpx.ASGITransport(app=small), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/analyze/stream"))
            async with asyncio.timeout(1):
                while not started.is_set():
                    await asyncio.sleep(0.001)
            assert (await client.get("/health")).status_code == 200
            assert (await client.post("/analyze/stream")).status_code == 503
            assert '"error"' in (await first).text
            assert cancelled.is_set()
            for _ in range(3):
                second = await client.post("/analyze/stream")
                assert second.status_code == 200 and '"success"' in second.text

    asyncio.run(check())


@pytest.mark.parametrize("cancelled", [False, True])
def test_capacity_is_released_when_request_raises_or_is_cancelled(cancelled):
    async def check():
        entered = asyncio.Event()
        calls = 0

        async def downstream(scope, receive, send):
            nonlocal calls
            calls += 1
            if calls == 1:
                entered.set()
                if cancelled:
                    await asyncio.sleep(3600)
                raise RuntimeError("upstream failed")

        app = AnalysisCapacity(downstream)
        scope = {"type": "http", "method": "POST", "path": "/analyze/stream"}

        async def unused(*args):
            raise AssertionError("unexpected ASGI message")

        first = asyncio.create_task(app(scope, unused, unused))
        await asyncio.wait_for(entered.wait(), 1)
        if cancelled:
            first.cancel()
        with pytest.raises(asyncio.CancelledError if cancelled else RuntimeError):
            await first
        await asyncio.wait_for(app(scope, unused, unused), 1)
        assert calls == 2

    asyncio.run(check())
