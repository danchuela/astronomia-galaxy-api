"""SIMBAD TAP: object metadata (type, morphology, radial velocity, redshift)."""

from __future__ import annotations

import logging
import os
from typing import Any

import requests

from packages.galaxy_core.infrastructure import availability

logger = logging.getLogger(__name__)

SIMBAD_TAP_URL = "https://simbad.cds.unistra.fr/simbad/sim-tap/sync"
SIMBAD_TAP_SERVICES = (
    ("simbad_cds", SIMBAD_TAP_URL),
    ("simbad_harvard", "https://simbad.cfa.harvard.edu/simbad/sim-tap/sync"),
    ("simbad_idia", "https://simbad.idia.ac.za/simbad/sim-tap/sync"),
)
CONNECT_TIMEOUT_SECONDS = 3
TIMEOUT_SECONDS = 6  # read timeout per service (was 15 s with a single service)


def query_object(name: str) -> dict[str, Any] | None:
    """Query SIMBAD TAP for object metadata (type, morph_type, redshift); returns dict or None."""
    name = name.strip()
    if not name:
        return None

    timeout_s = int(os.environ.get("SIMBAD_TIMEOUT", str(TIMEOUT_SECONDS)))

    adql = (
        "SELECT TOP 1 "
        "b.main_id, b.otype, b.otype_txt, b.morph_type, "
        "b.rvz_radvel, b.rvz_redshift, b.sp_type "
        "FROM basic AS b "
        "JOIN ident AS i ON i.oidref = b.oid "
        f"WHERE i.id = '{_escape_adql(name)}'"
    )

    params = {
        "request": "doQuery",
        "lang": "adql",
        "format": "json",
        "query": adql,
    }

    # The CDS first, then the SIMBAD mirrors (same database and TAP service). A mirror that
    # just failed is skipped for a while (availability.py) so a request does not wait for it.
    result = None
    for service, url in SIMBAD_TAP_SERVICES:
        if not availability.is_available(service):
            continue
        try:
            resp = requests.get(url, params=params, timeout=(CONNECT_TIMEOUT_SECONDS, timeout_s))
            resp.raise_for_status()
            result = resp.json()
            availability.mark_up(service)
            break
        except Exception:
            availability.mark_down(service)
            logger.warning(
                "simbad_query_failed",
                extra={"object_name": name, "service": service},
                exc_info=True,
            )
    if result is None:
        return None

    rows = result.get("data", [])
    columns = [c["name"] for c in result.get("metadata", [])]
    if not rows or not columns:
        return None

    row = rows[0]
    record: dict[str, Any] = {}
    for col_name, value in zip(columns, row, strict=True):
        key = col_name.lower()
        if value is not None and str(value).strip():
            record[key] = value

    return record if record.get("main_id") else None


def format_object_info(record: dict[str, Any], lang: str = "es") -> str:
    """Format SIMBAD record into a short description (Spanish or English) for the summary."""
    en = lang == "en"
    parts: list[str] = []

    main_id = record.get("main_id", "")
    otype_long = record.get("otype_txt") or record.get("otype") or ""
    if otype_long:
        parts.append(f"{main_id} — {'type' if en else 'tipo'}: {otype_long}")
    else:
        parts.append(str(main_id))

    morph = record.get("morph_type")
    if morph:
        parts.append(f"{'morphology' if en else 'morfología'}: {morph}")

    rv = record.get("rvz_radvel")
    if rv is not None:
        parts.append(f"{'radial velocity' if en else 'v_radial'}: {rv:.0f} km/s")

    z = record.get("rvz_redshift")
    if z is not None:
        parts.append(f"z = {z:.6f}")

    sp = record.get("sp_type")
    if sp:
        parts.append(f"{'spectral type' if en else 'tipo espectral'}: {sp}")

    return ". ".join(parts) + "." if parts else ""


def _escape_adql(value: str) -> str:
    """Escape single quotes for ADQL string literals."""
    return value.replace("'", "''")
