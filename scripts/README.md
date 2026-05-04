# scripts

Utility scripts for development, testing, and diagnostics. Run with `uv run python scripts/<name>.py`.

| Script | Purpose |
|--------|---------|
| **e2e_real.py** | End-to-end test against the live API. POSTs a natural language prompt, checks for success, and saves the result image to `artifacts/test-{HH}-{MM}/image.jpg`. Requires the API to be running. Configurable via `API_BASE_URL`. |
| **run_pipeline.py** | Runs the full analysis pipeline without the API: resolve target by name → fetch image → segment → measure. Useful for testing core logic in isolation. |
| **provider_probe.py** | Connectivity diagnostics against external image providers (HiPS, SkyView, SDSS, etc.). Tests DNS/TLS and downloads a sample image. Accepts `--ra`, `--dec`, `--fov-arcmin`, `--pixels`. Outputs JSON. |
| **debug_statmorph.py** | Runs statmorph on a downloaded galaxy image and prints all metrics to stdout. |
| **test_resolve_and_fetch.py** | Smoke test for name resolution (SESAME) and image download. Resolves a target by name, downloads an image from each active provider, and reports success/failure per provider. |
