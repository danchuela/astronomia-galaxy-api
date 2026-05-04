# astronomIA Galaxy API

Multi-band galaxy analysis API: target resolution by name or coordinates, image retrieval (optical, IR, UV, X-ray, radio), segmentation, morphological measurements, and LLM-generated summaries.

## Stack

| Layer | Package | Responsibility |
|-------|---------|----------------|
| HTTP | `apps/api` | FastAPI, routes, authentication, config |
| Agent | `packages/galaxy_agent` | Task orchestration, LangChain/OpenAI for NL, artifacts |
| Core | `packages/galaxy_core` | Segmentation, measurements, SESAME resolution, image catalogs, SIMBAD metadata, HST/JWST search |

### Image catalogs

Images are fetched through a provider registry with automatic fallback:

```
SDSS → DSS2 → GALEX → 2MASS → HiPS (20 surveys) → SkyView (fallback)
```

Direct clients (SDSS, DSS2, GALEX, 2MASS) take priority for their specific surveys. **HiPS** (`alasky.cds.unistra.fr`) covers the rest — PanSTARRS, DECaLS, WISE, RASS, XMM, NVSS and band variants.

Supported bands: `visible`/`optical`, `infrared`/`ir`, `ultraviolet`/`uv`, `x-ray`/`xray`, `radio`.

### Metadata enrichment

After resolving a target, the agent pipeline can query:

- **SIMBAD TAP** (CDS): object type, morphological type, radial velocity, redshift, spectral type. No API key required.
- **MAST HST/JWST** (STScI): checks for existing Hubble or JWST observations at the target position.

### Morphological analysis

See [`packages/galaxy_core/README.md`](packages/galaxy_core/README.md) for metric descriptions (CAS, Sérsic, radial profile, isophotes), value ranges, PSF limitations on ground-based images, and excluded metrics.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/)

## Running

```bash
cp .env.example .env   # set OPENAI_API_KEY at minimum
make install
make run               # uvicorn with --reload on :8000
```

**Docker:**

```bash
docker compose up --build
```

With GPU: `docker compose -f docker-compose.gpu.yml up --build`.

> In networks with corporate TLS inspection (e.g. Zscaler), place the root CA PEM at `docker/zscaler-root-ca.crt` before building (not tracked in git).

**Quick E2E:** with the API running, `python scripts/e2e_real.py`.

**Provider diagnostics:** `python scripts/provider_probe.py`.

## Tests and quality

```bash
make test        # pytest
make lint        # ruff
make format      # black
make typecheck   # mypy strict
make precommit   # all of the above
```

Single test:

```bash
uv run pytest tests/test_hips_strategy.py::test_panstarrs_resolves_via_hips -v
```

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| `GET` | `/health` | `{"status":"ok"}` |
| `POST` | `/analyze` | Synchronous analysis. Body: `request_id`, `target`, `task`, `image_url`, `options`, `message`/`messages` |
| `POST` | `/analyze/stream` | SSE analysis. Events: `status` → `summary` → `artifacts` → `end` (+ `error` on failure) |
| `GET` | `/artifacts/{request_id}/image` | Fetch generated/downloaded image |

The `artifacts` and `end` events include `coordinates` (ra_deg, dec_deg, survey_used, hips_id, size_arcmin), `object_info` (SIMBAD metadata), and `hst_jwst` (available HST/JWST observations).

Without API key in development: set `REQUIRE_API_KEY=false` in `.env`.

> Interactive docs (Swagger UI) at `http://localhost:8000/docs`.

## Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | OpenAI key (required for NL and agent) | — |
| `AGENT_MODEL` | LangGraph analysis agent model | `gpt-4.1` |
| `OPENAI_PARSE_MODEL` | NLU intent parser model | `gpt-4.1` |
| `OPENAI_MODEL` | Narrative summary generation model | `gpt-4.1-mini` |
| `REQUIRE_API_KEY` | Enable API key authentication | `true` |
| `API_KEY` | API key value | — |
| `ARTIFACT_DIR` | Artifact storage directory | `artifacts` |
| `LOG_LEVEL` | Log level | `INFO` |
| `SKYVIEW_TIMEOUT` | SkyView timeout (s) | `240` |
| `SESAME_TIMEOUT` | SESAME timeout (s) | `60` |
| `SIMBAD_TIMEOUT` | SIMBAD TAP timeout (s) | `15` |
| `LANGSMITH_API_KEY` | LangSmith key (optional) | — |
| `LANGSMITH_TRACING` | Enable LangSmith tracing | `false` |

See `.env.example` for the full list.
