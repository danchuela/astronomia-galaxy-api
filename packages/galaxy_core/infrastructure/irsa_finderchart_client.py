from __future__ import annotations

import io
import zipfile

import requests
from PIL import Image

FINDERCHART_API = "https://irsa.ipac.caltech.edu/applications/finderchart/servlet/api"


def fetch_2mass_cutout_jpeg(
    ra_deg: float,
    dec_deg: float,
    subsetsize_arcmin: float,
    timeout_sec: float = 60.0,
) -> bytes:
    params = {
        "mode": "getImage",
        "file_type": "png",
        "survey": "2mass",
        "subsetsize": str(float(subsetsize_arcmin)),
        "RA": str(float(ra_deg)),
        "DEC": str(float(dec_deg)),
        "reproject": "true",
        "marker": "false",
        "grid": "false",
    }
    resp = requests.get(
        FINDERCHART_API,
        params=params,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "application/zip" not in content_type and not resp.content.startswith(b"PK"):
        raise ValueError(
            f"FinderChart did not return a zip archive (content-type={content_type!r})."
        )

    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        png_names = [n for n in zf.namelist() if n.lower().endswith(".png")]
        if not png_names:
            raise ValueError("FinderChart zip contained no PNG files.")

        def score(name: str) -> tuple[int, str]:
            lower = name.lower()
            prefer_k = 0 if ("_k" in lower or "-k" in lower or " band k" in lower) else 1
            return (prefer_k, name)

        chosen = sorted(png_names, key=score)[0]
        png_bytes = zf.read(chosen)

    img = Image.open(io.BytesIO(png_bytes)).convert("RGB")
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()
