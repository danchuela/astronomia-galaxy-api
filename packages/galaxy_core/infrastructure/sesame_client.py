"""SESAME: resolve object name to J2000 (RA, Dec) in degrees."""

from __future__ import annotations

import asyncio
import math
import os
import re
import urllib.parse

import httpx
from astropy import units as u
from astropy.coordinates import SkyCoord

from packages.galaxy_core.domain.errors import TargetNotFoundError, TargetResolutionUnavailableError

SESAME_JPOS_RE = re.compile(r"^%J\s+([+-]?[\d.]+)\s+([+-]?[\d.]+)", re.MULTILINE)
SESAME_URL = "https://cds.unistra.fr/cgi-bin/nph-sesame"
SIMBAD_MIRROR_URL = "https://simbad.cfa.harvard.edu/simbad/sim-id"
TIMEOUT_SECONDS = 12.0
MAX_RESPONSE_BYTES = 128 * 1024
_ICRS_RE = re.compile(
    r"^Coordinates\(ICRS,ep=J2000,eq=2000\):\s*"
    r"(\d{1,2}\s+\d{1,2}\s+[\d.]+)\s+([+-]\d{1,2}\s+\d{1,2}\s+[\d.]+)",
    re.MULTILINE,
)


def _position(ra: float, dec: float) -> tuple[float, float]:
    if not (math.isfinite(ra) and math.isfinite(dec) and 0 <= ra < 360 and -90 <= dec <= 90):
        raise ValueError("Invalid resolver position")
    return ra, dec


def _parse_sesame(text: str, name: str) -> tuple[float, float]:
    match = SESAME_JPOS_RE.search(text)
    if match:
        return _position(float(match.group(1)), float(match.group(2)))
    for section in re.split(r"(?m)^# ", text):
        lines = section.splitlines()
        if lines and lines[0].split("\t", 1)[0].strip() == name:
            if re.search(r"nothing\s+found", "\n".join(lines[1:]), re.IGNORECASE):
                raise TargetNotFoundError(name)
    raise RuntimeError("SESAME returned no usable position or explicit not-found result")


def _parse_mirror(text: str) -> tuple[float, float]:
    matches = list(_ICRS_RE.finditer(text))
    if len(matches) != 1:
        # A SIMBAD-only negative answer cannot rule out NED/VizieR coverage.
        raise ValueError("SIMBAD mirror returned no unique J2000 position")
    match = matches[0]
    coordinates = SkyCoord(match.group(1), match.group(2), unit=(u.hourangle, u.deg), frame="icrs")
    return _position(float(coordinates.ra.deg), float(coordinates.dec.deg))


async def _read(client: httpx.AsyncClient, url: str) -> str:
    async with client.stream("GET", url) as response:
        response.raise_for_status()
        data = bytearray()
        async for chunk in response.aiter_bytes():
            data.extend(chunk)
            if len(data) > MAX_RESPONSE_BYTES:
                raise ValueError("Resolver response exceeds size limit")
        return data.decode(response.encoding or "utf-8", errors="replace")


async def _resolve_remote(name: str) -> tuple[float, float]:
    try:
        configured = float(os.environ.get("SESAME_TIMEOUT", str(TIMEOUT_SECONDS)))
        budget = (
            min(configured, TIMEOUT_SECONDS)
            if math.isfinite(configured) and configured > 0
            else TIMEOUT_SECONDS
        )
    except ValueError:
        budget = TIMEOUT_SECONDS
    timeout = httpx.Timeout(connect=2.0, read=4.0, write=2.0, pool=2.0)
    try:
        # Bound the whole lookup, including DNS, body reads and mirror fallback.
        async with asyncio.timeout(budget):
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
                try:
                    text = await _read(
                        client, f"{SESAME_URL}/SNV?{urllib.parse.quote(name, safe='')}"
                    )
                    return _parse_sesame(text, name)
                except TargetNotFoundError:
                    raise
                except (httpx.HTTPError, ValueError, RuntimeError):
                    query = urllib.parse.urlencode({"Ident": name, "output.format": "ASCII"})
                    return _parse_mirror(await _read(client, f"{SIMBAD_MIRROR_URL}?{query}"))
    except TargetNotFoundError:
        raise
    except (TimeoutError, httpx.HTTPError, ValueError, RuntimeError) as exc:
        raise TargetResolutionUnavailableError(name) from exc


def resolve(name: str) -> tuple[float, float]:
    name = name.strip()
    if not name:
        raise ValueError("Object name cannot be empty")

    # Messier designations are limited to M1-M110; no remote lookup is needed.
    # https://heasarc.gsfc.nasa.gov/W3Browse/all/messier.html
    messier = re.fullmatch(r"(?:M|Messier)\s*([0-9]+)", name, re.IGNORECASE)
    if messier and not 1 <= int(messier.group(1)) <= 110:
        raise TargetNotFoundError(name)

    return asyncio.run(_resolve_remote(name))
