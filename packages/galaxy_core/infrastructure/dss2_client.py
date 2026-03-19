"""DSS2 cutout client using STScI DSS service."""

from __future__ import annotations

import io
import urllib.parse

import requests
from PIL import Image

DSS2_SEARCH_URL = "https://archive.stsci.edu/cgi-bin/dss_search"

PLATE_RED = "poss2ukstu_red"
PLATE_BLUE = "poss2ukstu_blue"
PLATE_IR = "poss2ukstu_ir"
DEFAULT_PLATE = PLATE_RED


def survey_to_plate(survey: str) -> str | None:
    normalized = " ".join(survey.strip().upper().replace("-", " ").replace("_", " ").split())
    if not normalized:
        return None

    if normalized in ("DSS", "DSS2"):
        return PLATE_RED

    if normalized.startswith("DSS2") or normalized.startswith("POSS2UKSTU"):
        tokens = set(normalized.split())
        if "BLUE" in tokens:
            return PLATE_BLUE
        if "IR" in tokens or "INFRARED" in tokens:
            return PLATE_IR
        return PLATE_RED

    return None


def get_image_url(
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float = 10.0,
    plate: str = DEFAULT_PLATE,
) -> str:
    ra = float(ra_deg) % 360.0
    dec = max(-90.0, min(90.0, float(dec_deg)))
    size = max(0.1, min(60.0, float(size_arcmin)))

    params = {
        "v": plate,
        "r": round(ra, 6),
        "d": round(dec, 6),
        "e": "J2000",
        "h": round(size, 4),
        "w": round(size, 4),
        "f": "gif",
        "c": "none",
    }
    return f"{DSS2_SEARCH_URL}?{urllib.parse.urlencode(params)}"


def fetch_cutout_jpeg(
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
    plate: str = DEFAULT_PLATE,
    timeout_sec: float = 60.0,
) -> bytes:
    url = get_image_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        size_arcmin=size_arcmin,
        plate=plate,
    )
    response = requests.get(
        url,
        timeout=timeout_sec,
    )
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").lower()
    if "image/gif" not in content_type and not response.content.startswith(b"GIF8"):
        raise ValueError(
            f"DSS2 endpoint did not return a GIF image (content-type={content_type!r})."
        )

    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92, optimize=True)
    return out.getvalue()
