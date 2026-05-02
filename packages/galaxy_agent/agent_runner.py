from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.domain.models import TaskType
from packages.galaxy_agent.langchain_backend import LangChainBackend
from packages.galaxy_agent.messages import (
    ERROR_ANALYSIS_FAILED_MESSAGE,
    ERROR_IMAGE_FETCH_FAILED_MESSAGE,
    TARGET_REQUIRED_MESSAGE,
)
from packages.galaxy_agent.models import (
    AnalyzeRequest,
    AnalyzeResponse,
    Target,
)
from packages.galaxy_agent.orchestrator import TaskOrchestrator
from packages.galaxy_agent.provenance_utils import (
    build_provenance,
    build_stream_provenance_payload,
)
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer

logger = logging.getLogger(__name__)

_DEFAULT_TASK: TaskType = "morphology_summary"
_PLACEHOLDER_TARGET = "from conversation"


class AgentRunner:
    def __init__(
        self,
        artifact_dir: str = "artifacts",
        langsmith_enabled: bool = False,
    ) -> None:
        self.langsmith_enabled = langsmith_enabled
        self.analyzer = BasicGalaxyAnalyzer()
        self.langchain_backend = LangChainBackend()
        self.orchestrator = TaskOrchestrator(
            analyzer=self.analyzer,
            artifact_store=ArtifactStore(artifact_dir),
            langchain_backend=self.langchain_backend,
        )

    def run(self, request: AnalyzeRequest) -> AnalyzeResponse:
        try:
            enriched = self.langchain_backend.enrich_request(request)
            if enriched.out_of_scope and enriched.decline_message:
                return self._build_success_response(
                    request_id=request.request_id,
                    summary=enriched.decline_message,
                )
            resolved = self._resolve_request(enriched)
            if self._needs_target_prompt(resolved):
                return self._build_success_response(
                    request_id=request.request_id,
                    summary=TARGET_REQUIRED_MESSAGE,
                )

            return self.orchestrator.run(request=resolved, langsmith_enabled=self.langsmith_enabled)
        except Exception:
            logger.exception(
                "analysis_failed",
                extra={"request_id": request.request_id, "event": "error"},
            )
            return self._build_error_response(
                request_id=request.request_id,
                summary=ERROR_ANALYSIS_FAILED_MESSAGE,
            )

    def _resolve_request(self, request: AnalyzeRequest) -> AnalyzeRequest:
        if request.target is not None and request.task is not None:
            return request
        # When a viewer snapshot is present, skip the target placeholder — the
        # orchestrator reads view_ra_deg/view_hips_id directly
        if request.target is None and request.view_ra_deg is not None:
            target = None
        else:
            target = request.target or Target(name=_PLACEHOLDER_TARGET)
        task: TaskType = request.task if request.task is not None else _DEFAULT_TASK
        return request.to_resolved_request(target=target, task=task)  # type: ignore[arg-type]

    def _needs_target_prompt(self, request: AnalyzeRequest) -> bool:
        opts = request.options or {}
        has_coords = opts.get("ra_deg") is not None
        has_view = request.view_ra_deg is not None
        return bool(
            request.target
            and (request.target.name or "").strip() in ("", _PLACEHOLDER_TARGET)
            and not request.image_url
            and not has_coords
            and not has_view
        )

    def _build_success_response(self, request_id: str, summary: str) -> AnalyzeResponse:
        return AnalyzeResponse(
            request_id=request_id,
            status="success",
            summary=summary,
            results={},
            artifacts=[],
            provenance=build_provenance(self.langsmith_enabled),
            warnings=[],
        )

    def _build_error_response(self, request_id: str, summary: str) -> AnalyzeResponse:
        return AnalyzeResponse(
            request_id=request_id,
            status="error",
            summary=summary,
            results={},
            artifacts=[],
            provenance=build_provenance(self.langsmith_enabled),
            warnings=[],
        )

    def _build_stream_end_event(self, request_id: str, status: str, summary: str) -> dict[str, Any]:
        return {
            "type": "end",
            "request_id": request_id,
            "status": status,
            "summary": summary,
            "results": {},
            "artifacts": [],
            "provenance": build_stream_provenance_payload(self.langsmith_enabled),
            "warnings": [],
        }

    def run_stream(self, request: AnalyzeRequest) -> Iterator[dict[str, Any]]:
        try:
            enriched = self.langchain_backend.enrich_request(request)
            if enriched.out_of_scope and enriched.decline_message:
                yield {"type": "summary", "summary": enriched.decline_message}
                yield self._build_stream_end_event(
                    request_id=request.request_id,
                    status="success",
                    summary=enriched.decline_message,
                )
                return
            resolved = self._resolve_request(enriched)
            if self._needs_target_prompt(resolved):
                yield {"type": "summary", "summary": TARGET_REQUIRED_MESSAGE}
                yield self._build_stream_end_event(
                    request_id=request.request_id,
                    status="success",
                    summary=TARGET_REQUIRED_MESSAGE,
                )
                return

            yield from self.orchestrator.run_stream(
                request=resolved, langsmith_enabled=self.langsmith_enabled
            )
        except Exception as e:
            logger.exception(
                "analysis_failed",
                extra={"request_id": request.request_id, "event": "error"},
            )
            message = ERROR_ANALYSIS_FAILED_MESSAGE
            err_str = str(e)
            if (
                "Failed to resolve and fetch image" in err_str
                or "Image download failed for survey" in err_str
            ):
                message = ERROR_IMAGE_FETCH_FAILED_MESSAGE
            yield {"type": "summary", "summary": message}
            yield self._build_stream_end_event(
                request_id=request.request_id,
                status="error",
                summary=message,
            )
            return
