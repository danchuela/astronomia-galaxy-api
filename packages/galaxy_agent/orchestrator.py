from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.messages import HumanMessage

from packages.galaxy_agent.agent_tools import build_agent_tools
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.galaxy_agent import build_galaxy_agent
from packages.galaxy_agent.langchain_backend import LangChainBackend
from packages.galaxy_agent.models import AnalyzeRequest, AnalyzeResponse, Artifact, Provenance
from packages.galaxy_agent.provenance_utils import build_provenance
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer

logger = logging.getLogger(__name__)


class TaskOrchestrator:
    def __init__(
        self,
        analyzer: BasicGalaxyAnalyzer,
        artifact_store: ArtifactStore,
        langchain_backend: LangChainBackend | None = None,
    ) -> None:
        self.analyzer = analyzer
        self.artifact_store = artifact_store
        self.langchain_backend = langchain_backend

    def run(self, request: AnalyzeRequest, langsmith_enabled: bool) -> AnalyzeResponse:
        for event in self.run_stream(request, langsmith_enabled):
            if event.get("type") == "end":
                artifacts_d = event.get("artifacts", [])
                prov = event.get("provenance")
                return AnalyzeResponse(
                    request_id=event["request_id"],
                    status=event["status"],
                    summary=event.get("summary", ""),
                    results=event.get("results", {}),
                    artifacts=[Artifact(**a) for a in artifacts_d],
                    provenance=(
                        Provenance(**prov)
                        if isinstance(prov, dict)
                        else prov if prov is not None else build_provenance(langsmith_enabled)
                    ),
                    warnings=event.get("warnings", []),
                )
            if event.get("type") == "error":
                raise RuntimeError(event.get("message", "Unknown error"))
        raise RuntimeError("Stream ended without end event")

    def run_stream(
        self, request: AnalyzeRequest, langsmith_enabled: bool
    ) -> Iterator[dict[str, Any]]:
        warnings: list[str] = []
        registry = ContextRegistry(request.request_id)
        rid = request.request_id

        _TOOL_STATUS: dict[str, str] = {
            "resolve_and_fetch_image": "Resolviendo objetivo…",
            "segment_image": "Segmentando imagen…",
            "analyze_galaxy": "Calculando análisis morfológico…",
            "run_isophotes": "Ajustando isofotas elípticas…",
            "enrich_metadata": "Consultando catálogos externos…",
            "generate_final_report": "Generando resumen…",
        }

        try:
            tools = build_agent_tools(
                registry=registry,
                analyzer=self.analyzer,
                artifact_store=self.artifact_store,
                request=request,
                langchain_backend=self.langchain_backend,
            )
            agent = build_galaxy_agent(tools)

            opts = request.options or {}
            has_viewer = request.view_ra_deg is not None
            target_name = (request.target and request.target.name) or "none"
            context_lines = [
                f"request_id: {rid}",
                f"task: {request.task or 'resolve'}",
                f"target: {target_name}",
                f"has_viewer_image: {has_viewer}",
                f"has_image_data: {bool(request.image_data)}",
            ]
            if has_viewer:
                context_lines.append(f"view_hips_id: {request.view_hips_id or 'none'}")
            if opts.get("band"):
                context_lines.append(f"band: {opts['band']}")
            if opts.get("catalog"):
                context_lines.append(f"catalog: {opts['catalog']}")

            human_msg = HumanMessage(content="\n".join(context_lines))

            for chunk in agent.stream({"messages": [human_msg]}, stream_mode="updates"):
                if "model" in chunk:
                    for msg in chunk["model"].get("messages", []):
                        for tc in getattr(msg, "tool_calls", []):
                            status_msg = _TOOL_STATUS.get(tc.get("name", ""))
                            if status_msg:
                                yield {"type": "status", "message": status_msg}

        except Exception as exc:
            logger.error(
                "agent_run_error",
                extra={"request_id": rid, "error": str(exc)},
                exc_info=True,
            )
            yield {"type": "error", "message": str(exc)}
            registry.clear()
            return

        # Read accumulated state from registry and assemble final SSE events
        def _safe_get(key: str) -> Any:
            try:
                return registry.get(key)
            except KeyError:
                return None

        summary: str = _safe_get(f"summary:{rid}") or ""
        artifacts_list: list[Artifact] = _safe_get(f"artifacts:{rid}") or []
        results: dict[str, Any] = _safe_get(f"results:{rid}") or {}
        coordinates: dict[str, Any] | None = _safe_get(f"coordinates:{rid}")
        object_info: Any = _safe_get(f"object_info:{rid}")
        hst_jwst: Any = _safe_get(f"hst_jwst:{rid}")
        object_name = (request.target and request.target.name) or None

        if not summary:
            logger.warning(
                "agent_empty_summary",
                extra={"request_id": rid, "task": request.task},
            )
            target_label = (request.target and request.target.name) or "el objeto"
            if results:
                summary = (
                    f"Análisis de {target_label} completado. "
                    f"Consulta los artefactos para ver los resultados."
                )
            elif coordinates:
                ra = coordinates.get("ra_deg")
                dec = coordinates.get("dec_deg")
                label: str = (
                    target_label
                    if request.target
                    else (f"RA {ra:.4f}° Dec {dec:.4f}°" if ra and dec else "el objeto")
                )
                summary = (
                    f"Aquí tienes {label}. "
                    f"Ajusta el encuadre, el zoom y la banda en el visor como prefieras. "
                    f"Cuando estés listo, dime qué quieres analizar."
                )
            else:
                registry.clear()
                yield {
                    "type": "error",
                    "message": "El agente completó la ejecución sin generar un resumen.",
                }
                return

        registry.clear()

        logger.info(
            "analysis_completed",
            extra={"request_id": rid, "task": request.task, "event": "analysis"},
        )

        response = self._build_response(
            request, summary, results, artifacts_list, warnings, langsmith_enabled
        )

        yield {"type": "summary", "summary": response.summary}

        if artifacts_list:
            artifacts_event: dict[str, Any] = {
                "type": "artifacts",
                "request_id": rid,
            }
            plot_paths = [a.path for a in artifacts_list if a.type == "plot"]
            if plot_paths:
                artifacts_event["analysis_plots"] = [
                    p.split("/")[-1].removeprefix("plot-").removesuffix(".png") for p in plot_paths
                ]
            if coordinates:
                artifacts_event["coordinates"] = coordinates
            if object_info:
                artifacts_event["object_info"] = object_info
            if hst_jwst:
                artifacts_event["hst_jwst"] = hst_jwst
            if object_name:
                artifacts_event["object_name"] = object_name
            yield artifacts_event

        end_event: dict[str, Any] = {
            "type": "end",
            "request_id": response.request_id,
            "status": response.status,
            "summary": response.summary,
            "results": response.results,
            "artifacts": [a.model_dump() for a in response.artifacts],
            "provenance": response.provenance.model_dump(),
            "warnings": response.warnings,
        }
        if coordinates:
            end_event["coordinates"] = coordinates
        if object_info:
            end_event["object_info"] = object_info
        if hst_jwst:
            end_event["hst_jwst"] = hst_jwst
        if object_name:
            end_event["object_name"] = object_name
        yield end_event

    def _build_response(
        self,
        request: AnalyzeRequest,
        summary: str,
        results: dict[str, Any],
        artifacts: list[Artifact],
        warnings: list[str],
        langsmith_enabled: bool,
    ) -> AnalyzeResponse:
        provenance = build_provenance(langsmith_enabled)
        return AnalyzeResponse(
            request_id=request.request_id,
            status="success",
            summary=summary,
            results=results,
            artifacts=artifacts,
            provenance=provenance,
            warnings=warnings,
        )
