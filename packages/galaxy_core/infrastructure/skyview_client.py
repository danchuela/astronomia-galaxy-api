"""SkyView image URL for (RA, Dec) and survey."""

from __future__ import annotations

import os
import re

import requests

SKYVIEW_RUNQUERY_URL = "https://skyview.gsfc.nasa.gov/current/cgi/runquery.pl"
TIMEOUT_SECONDS = 240
DEFAULT_PIXELS = 300


def get_image_url(
    ra_deg: float,
    dec_deg: float,
    survey: str,
    size_arcmin: float = 10.0,
) -> str:
    timeout_s = int(os.environ.get("SKYVIEW_TIMEOUT", str(TIMEOUT_SECONDS)))
    scale_arcsec = (size_arcmin * 60.0) / DEFAULT_PIXELS
    position = f"{ra_deg},{dec_deg}"

    data = {
        "Position": position,
        "Survey": survey,
        "Pixels": str(DEFAULT_PIXELS),
        "Scale": str(round(scale_arcsec, 4)),
        "Coordinates": "J2000",
    }

    base = "https://skyview.gsfc.nasa.gov"
    url_pattern = re.compile(
        r'href="(https?://[^"]*skyview[^"]*\.nasa\.gov[^"]+\.(?:fits|fits\.gz|jpg|jpeg|png))"',
        re.IGNORECASE,
    )
    rel_pattern = re.compile(r'href="(/[^"]+\.(?:fits|fits\.gz|jpg|jpeg|png))"', re.IGNORECASE)
    img_src_pattern = re.compile(
        r'src="(https?://[^"]*skyview[^"]*\.nasa\.gov[^"]+\.(?:jpg|jpeg|png))"',
        re.IGNORECASE,
    )

    with requests.post(
        SKYVIEW_RUNQUERY_URL,
        data=data,
        timeout=timeout_s,
        headers={"User-Agent": "astronomIA-galaxy-api/1.0"},
        stream=True,
    ) as response:
        response.raise_for_status()
        buffer = ""
        for chunk in response.iter_content(chunk_size=8192):
            if not chunk:
                continue
            buffer += chunk.decode("utf-8", errors="replace")
            match = url_pattern.search(buffer)
            if match:
                return match.group(1)
            rel_match = rel_pattern.search(buffer)
            if rel_match:
                return base + rel_match.group(1)
            img_match = img_src_pattern.search(buffer)
            if img_match:
                return img_match.group(1)

    raise ValueError(
        f"No image URL found in SkyView response for survey={survey} at ({ra_deg}, {dec_deg})"
    )
