from __future__ import annotations

import json
import logging
import os
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from dotenv import load_dotenv

load_dotenv(override=False)

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from apps.api.auth import verify_api_key
from apps.api.config import Settings, get_settings
from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.logging_utils import setup_logging
from packages.galaxy_agent.models import AnalyzeRequest, AnalyzeResponse


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger.info("api_started", extra={"event": "startup"})
    logger.info(
        "openai_check",
        extra={
            "event": "startup",
            "openai_configured": bool((os.getenv("OPENAI_API_KEY") or "").strip()),
        },
    )
    yield


app = FastAPI(title="astronomIA Galaxy API", version="0.1.0", lifespan=lifespan)
logger = logging.getLogger(__name__)


def get_runner(settings: Annotated[Settings, Depends(get_settings)]) -> AgentRunner:
    langsmith_enabled = bool(settings.langsmith_api_key) or settings.langsmith_tracing
    return AgentRunner(
        artifact_dir=settings.artifact_dir,
        langsmith_enabled=langsmith_enabled,
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse, dependencies=[Depends(verify_api_key)])
def analyze(
    request: AnalyzeRequest,
    runner: Annotated[AgentRunner, Depends(get_runner)],
) -> AnalyzeResponse:
    return runner.run(request)


def _sse_stream(request: AnalyzeRequest, runner: AgentRunner) -> Iterator[bytes]:
    try:
        for event in runner.run_stream(request):
            event_type = event.get("type", "status")
            line = f"event: {event_type}\ndata: {json.dumps(event)}\n\n"
            yield line.encode("utf-8")
    except Exception:
        logger.exception("sse_stream_error")
        error_event = {
            "type": "end",
            "status": "error",
            "summary": "Internal stream error.",
        }
        line = f"event: end\ndata: {json.dumps(error_event)}\n\n"
        yield line.encode("utf-8")


@app.post("/analyze/stream", dependencies=[Depends(verify_api_key)])
def analyze_stream(
    request: AnalyzeRequest,
    runner: Annotated[AgentRunner, Depends(get_runner)],
) -> StreamingResponse:
    return StreamingResponse(
        _sse_stream(request, runner),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/artifacts/{request_id}/image", dependencies=[Depends(verify_api_key)])
def get_artifact_image(
    request_id: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    base = Path(settings.artifact_dir).resolve()
    path = (base / request_id / "image.jpg").resolve()
    if not path.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid request_id.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Image not found for this request_id.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/artifacts/{request_id}/plot/{plot_name}", dependencies=[Depends(verify_api_key)])
def get_artifact_plot(
    request_id: str,
    plot_name: str,
    settings: Annotated[Settings, Depends(get_settings)],
) -> FileResponse:
    base = Path(settings.artifact_dir).resolve()
    safe_name = plot_name.replace("/", "_").replace("..", "_")
    path = (base / request_id / f"plot-{safe_name}.png").resolve()
    if not path.is_relative_to(base):
        raise HTTPException(status_code=400, detail="Invalid request_id or plot_name.")
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(path, media_type="image/png")
