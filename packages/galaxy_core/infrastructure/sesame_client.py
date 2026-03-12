"""SESAME: resolve object name to J2000 (RA, Dec) in degrees."""

from __future__ import annotations

import os
import re
import urllib.parse

import requests

SESAME_JPOS_RE = re.compile(r"^%J\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)", re.MULTILINE)
SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame"
TIMEOUT_SECONDS = 60


def resolve(name: str) -> tuple[float, float]:
    name = name.strip()
    if not name:
        raise ValueError("Object name cannot be empty")

    timeout_s = int(os.environ.get("SESAME_TIMEOUT", str(TIMEOUT_SECONDS)))
    url = f"{SESAME_URL}?-ox&{urllib.parse.quote(name)}"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    text = response.text

    match = SESAME_JPOS_RE.search(text)
    if not match:
        raise ValueError(f"SESAME returned no position for '{name}'")

    ra_deg = float(match.group(1))
    dec_deg = float(match.group(2))
    return (ra_deg, dec_deg)
