# Architecture and development guide

## Overview

Three layers; each layer only imports inward:

1. **API** (`apps/api`) — HTTP entry point; delegates to the agent.
2. **Agent** (`packages/galaxy_agent`) — Task orchestration, LangGraph ReAct agent with GPT-4.1 tool-calling, LangChain/OpenAI for NL parsing, artifact storage.
3. **Core** (`packages/galaxy_core`) — Segmentation, morphological measurements, image fetching. No LLM dependency.

Request flow:

```
Client  →  POST /analyze/stream  →  apps.api.main  →  AgentRunner.run()
                                                     →  LangChainBackend (NL → target/task/band)
                                                     →  TaskOrchestrator.run_stream()
                                                          →  LangGraph agent
                                                               →  agent_tools (resolve/fetch, segment, analyze, enrich, report)
                                                                    →  BasicGalaxyAnalyzer (galaxy_core)
                                                                    →  survey_provider_registry
                                                                    →  ArtifactStore (images, masks, plots, measurements)
                                                     →  SSE: status → summary → artifacts → end
```

---

## File tree and responsibilities

```
astronomIA-galaxy-api/
├── apps/
│   └── api/
│       ├── main.py             # FastAPI app, routes /health /analyze /analyze/stream /artifacts
│       ├── config.py           # Settings from env (API_KEY, ARTIFACT_DIR, LOG_LEVEL…)
│       └── auth.py             # X-API-Key verification (timing-safe)
│
├── packages/
│   ├── galaxy_core/
│   │   ├── domain/             # imaging.py (BAND_TO_SURVEY), models.py (SegmentationResult, GalaxyAnalyzer)
│   │   ├── application/
│   │   │   ├── analyzer_service.py       # BasicGalaxyAnalyzer: segment, run_task, measure_basic
│   │   │   ├── resolve_and_fetch_service.py  # Target resolution + image download
│   │   │   ├── morphology.py             # SEP background, segmap, statmorph metrics, radial profile
│   │   │   ├── isophotes.py              # Ellipse fitting + isointensity contour rendering
│   │   │   └── analyses/                # Registry + per-module analysis classes (CAS, Sérsic, RadialProfile)
│   │   ├── infrastructure/     # External clients and provider registry
│   │   └── analyzer.py         # Public re-exports: BasicGalaxyAnalyzer, create_synthetic_image
│   │
│   └── galaxy_agent/
│       ├── domain/             # models.py: AnalyzeRequest, AnalyzeResponse, Artifact, Provenance, TaskType
│       ├── agent_runner.py     # Entry point: run(request) → response; enrich + orchestrate
│       ├── orchestrator.py     # Builds per-request agent, streams status/SSE events
│       ├── agent_tools.py      # LangChain tool factories for image acquisition and analysis
│       ├── tools.py            # Thin adapters: load_image, tool_segment, tool_run_analysis, tool_isophotes
│       ├── artifacts.py        # ArtifactStore: images, masks, plots, measurements in artifacts/<request_id>/
│       ├── langchain_backend.py # NL parsing (message → target/task), summary generation
│       ├── logging_utils.py    # JsonFormatter, setup_logging
│       ├── messages.py         # Error/UX messages
│       ├── request_utils.py    # last_user_message, append_catalog_and_field
│       ├── provenance_utils.py # Version stamps and timestamps
│       └── constants.py        # GALAXY_CORE_VERSION, GALAXY_AGENT_VERSION
│
├── tests/
├── scripts/                    # See scripts/README.md
├── docker/
├── pyproject.toml
├── Makefile
└── .env.example
```

---

## Infrastructure: image providers

`survey_provider_registry.py` implements a strategy pattern with automatic fallback:

```
SDSS → DSS2 → GALEX → 2MASS → HiPS → SkyView (fallback)
```

| Client | Module | Surveys |
|--------|--------|---------|
| SDSS | `sdss_client.py` | SDSS DR18 |
| DSS2 | `dss2_client.py` | DSS2 red/blue/IR |
| GALEX | `mast_galex_client.py` | GALEX NUV/FUV via MAST |
| 2MASS | `irsa_finderchart_client.py` | 2MASS J/H/K via IRSA |
| HiPS | `hips_client.py` | 20 surveys via `alasky.cds.unistra.fr/hips2fits` |
| SkyView | `skyview_client.py` | NASA SkyView URLs (broadest coverage, highest latency) |

**SIMBAD TAP** (`simbad_client.py`): object type, morphological type, radial velocity, redshift, spectral type. No API key required.

**MAST HST/JWST** (`mast_hst_client.py`): checks whether HST or JWST observations exist for a position.

**SESAME** (`sesame_client.py`): resolves astronomical names (M51, NGC 1300) to RA/Dec via CDS SESAME.

---

## Task types

| Task | Description |
|------|-------------|
| `resolve` | SESAME name resolution only — returns coordinates for the viewer, no analysis |
| `segment` | SEP segmentation + contour overlay on the image |
| `measure_basic` | Segmentation + photometry (area, centroid, ellipticity, intensity, CAS, Sérsic, radial profile, isophotes) |
| `morphology_summary` | Full pipeline: segment + CAS + Sérsic + radial profile + isophotes + LLM narrative |
| `cas` | CAS indices only |
| `radial_profile` | Radial brightness profile only |
| `sersic` | Sérsic profile fit only |
| `isophotes` | Elliptical isophote fitting + contour rendering |

For metric descriptions and PSF caveats see `packages/galaxy_core/README.md`.

---

## Where to develop what

| Goal | Where to touch |
|------|----------------|
| New endpoints | `apps/api/main.py` |
| New config / env vars | `apps/api/config.py` and `.env.example` |
| New analysis logic | `packages/galaxy_core/application/analyses/` |
| New tasks or flows | `packages/galaxy_agent/galaxy_agent.py`, `agent_tools.py`, and `tools.py` |
| LLM integration | `packages/galaxy_agent/langchain_backend.py` |
| Request/response schema | `packages/galaxy_agent/domain/models.py` |
| Artifact storage | `packages/galaxy_agent/artifacts.py` |
| New survey | `packages/galaxy_core/infrastructure/hips_client.py` + `domain/imaging.py` |

**Dependency rule:** `galaxy_core` never imports LangChain or FastAPI; `galaxy_agent` imports core and LangChain; `apps/api` only imports from `galaxy_agent`.

---

## Adding a new analysis task

1. Add the task name to `TaskType` in `packages/galaxy_agent/domain/models.py`.
2. Create a module class implementing `run(ctx: AnalysisContext) -> AnalysisResult` in `packages/galaxy_core/application/analyses/`. Register it in `registry.py` and add it to `TASK_MODULES`.
3. Update the tool execution rules in `packages/galaxy_agent/galaxy_agent.py` if the new task changes the tool sequence.
4. Add the task description to the LLM prompt in `packages/galaxy_agent/langchain_backend.py`.
5. Add focused tests for the module and agent tool path.

## Adding a new survey

1. Add the HiPS ID in `SURVEY_TO_HIPS` in `hips_client.py`.
2. If it's a new band, add an alias in `BAND_TO_SURVEY` in `domain/imaging.py`.
3. Add a test in `tests/test_hips_strategy.py`.
