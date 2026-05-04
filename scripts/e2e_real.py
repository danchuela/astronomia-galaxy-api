#!/usr/bin/env python3
"""E2E: POST /analyze with NL prompt — verify success and artifact image on disk.

Requires: API running (docker compose up), OPENAI_API_KEY in .env.

  python scripts/e2e_real.py
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE = os.environ.get("API_BASE_URL", "http://localhost:8000")
API_KEY = os.environ.get("API_KEY", "")
ARTIFACT_DIR = os.environ.get("ARTIFACT_DIR", "artifacts")


def main() -> None:
    now = datetime.now()
    request_id = f"test-{now.hour:02d}-{now.minute:02d}"
    body = {"request_id": request_id, "message": "Dame una imagen de M104 en visible"}

    project_root = Path(__file__).resolve().parent.parent
    image_path = project_root / ARTIFACT_DIR / request_id / "image.jpg"

    print("E2E:", BASE, "|", body)
    print()

    try:
        with urlopen(f"{BASE}/health", timeout=5) as r:
            if r.status != 200:
                raise HTTPError(r.url, r.status, r.reason, r.headers, r)
    except (URLError, HTTPError, OSError) as e:
        print("ERROR: API no responde. ¿docker compose up?", e)
        sys.exit(1)

    body_bytes = json.dumps(body).encode("utf-8")
    req = Request(
        f"{BASE}/analyze",
        data=body_bytes,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)

    print("POST /analyze (30–60 s)...")
    try:
        with urlopen(req, timeout=120) as r:
            if r.status != 200:
                raise HTTPError(r.url, r.status, r.reason, r.headers, r)
            data = json.loads(r.read().decode())
    except (URLError, HTTPError, OSError) as e:
        print("ERROR /analyze:", e)
        sys.exit(1)

    if data.get("status") != "success":
        print("ERROR status:", data.get("status"), "|", data.get("summary", ""))
        sys.exit(1)

    if not image_path.exists():
        print("ERROR: imagen no encontrada en", image_path)
        sys.exit(1)

    print("OK")
    print(f"  summary: {data.get('summary', '')[:100]}...")
    print(f"  image:   {image_path}")


if __name__ == "__main__":
    main()
