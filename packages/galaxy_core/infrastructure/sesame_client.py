"""SESAME: resolve object name to J2000 (RA, Dec) in degrees."""

from __future__ import annotations

import os
import re
import urllib.parse

import requests

from packages.galaxy_core.domain.errors import TargetNotFoundError

SESAME_JPOS_RE = re.compile(r"^%J\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)", re.MULTILINE)
SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame"
TIMEOUT_SECONDS = 60


def resolve(name: str) -> tuple[float, float]:
    name = name.strip()
    if not name:
        raise ValueError("Object name cannot be empty")

    # Messier designations are limited to M1-M110; no remote lookup is needed.
    # https://heasarc.gsfc.nasa.gov/W3Browse/all/messier.html
    messier = re.fullmatch(r"(?:M|Messier)\s*([0-9]+)", name, re.IGNORECASE)
    if messier and not 1 <= int(messier.group(1)) <= 110:
        raise TargetNotFoundError(name)

    timeout_s = int(os.environ.get("SESAME_TIMEOUT", str(TIMEOUT_SECONDS)))
    url = f"{SESAME_URL}?-ox&{urllib.parse.quote(name)}"
    response = requests.get(url, timeout=timeout_s)
    response.raise_for_status()
    text = response.text

    match = SESAME_JPOS_RE.search(text)
    if not match:
        if not text.strip() or "<html" in text.lower():
            raise RuntimeError("SESAME returned an invalid response")
        raise TargetNotFoundError(name)

    ra_deg = float(match.group(1))
    dec_deg = float(match.group(2))
    return (ra_deg, dec_deg)
