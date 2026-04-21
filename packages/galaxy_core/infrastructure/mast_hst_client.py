"""MAST: search for HST/JWST observations and build cutout URLs."""

from __future__ import annotations

import json
import logging
import sys
import urllib.parse
from typing import Any

import requests

logger = logging.getLogger(__name__)

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"
TIMEOUT_SEC = 30.0

_TARGET_COLLECTIONS = ("HST", "JWST")


def _mast_invoke(payload: dict[str, Any], timeout_sec: float = TIMEOUT_SEC) -> dict[str, Any]:
    version = ".".join(map(str, sys.version_info[:3]))
    headers = {
        "Content-type": "application/x-www-form-urlencoded",
        "Accept": "text/plain",
        "User-agent": f"python-requests/{version}",
    }
    req_string = urllib.parse.quote(json.dumps(payload))
    resp = requests.post(
        MAST_INVOKE_URL,
        data=f"request={req_string}",
        headers=headers,
        timeout=timeout_sec,
    )
    resp.raise_for_status()
    result: dict[str, Any] = json.loads(resp.content.decode("utf-8"))
    return result


def search_hst_jwst(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float = 0.05,
) -> dict[str, Any] | None:
    """Search MAST for HST/JWST imaging at a position; returns observation dict or None."""
    for collection in _TARGET_COLLECTIONS:
        result = _search_collection(ra_deg, dec_deg, radius_deg, collection)
        if result is not None:
            return result
    return None


def _search_collection(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    collection: str,
) -> dict[str, Any] | None:
    payload = {
        "service": "Mast.Caom.Filtered.Position",
        "format": "json",
        "params": {
            "columns": "obs_collection,obs_id,instrument_name,filters,jpegURL,dataproduct_type",
            "filters": [
                {"paramName": "obs_collection", "values": [collection]},
                {"paramName": "dataproduct_type", "values": ["image"]},
            ],
            "position": f"{ra_deg}, {dec_deg}, {radius_deg}",
        },
        "pagesize": 20,
        "page": 1,
        "removenullcolumns": True,
        "removecache": True,
    }

    try:
        out = _mast_invoke(payload)
    except Exception:
        logger.warning(
            "mast_hst_search_failed",
            extra={"collection": collection, "ra": ra_deg, "dec": dec_deg},
            exc_info=True,
        )
        return None

    data = out.get("data") or []
    best: dict[str, Any] | None = None
    for row in data:
        jpeg_url = row.get("jpegURL")
        obs_id = row.get("obs_id", "")
        if isinstance(jpeg_url, str) and jpeg_url.strip():
            return {
                "collection": collection,
                "obs_id": obs_id,
                "instrument": row.get("instrument_name", ""),
                "filters": row.get("filters", ""),
                "jpeg_url": jpeg_url.strip(),
                "dataproduct_type": row.get("dataproduct_type", "image"),
            }
        if best is None and obs_id:
            best = {
                "collection": collection,
                "obs_id": obs_id,
                "instrument": row.get("instrument_name", ""),
                "filters": row.get("filters", ""),
                "jpeg_url": None,
                "dataproduct_type": row.get("dataproduct_type", "image"),
            }

    return best


def format_hst_jwst_info(record: dict[str, Any]) -> str:
    """Format MAST HST/JWST result into a short Spanish note."""
    coll = record.get("collection", "")
    instrument = record.get("instrument", "")
    filters = record.get("filters", "")

    label = coll
    if instrument:
        label += f"/{instrument}"
    if filters:
        label += f" ({filters})"

    return f"Observación disponible en {label}."
