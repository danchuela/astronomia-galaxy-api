from __future__ import annotations

import json
import sys
import urllib.parse
from typing import Any

import requests

MAST_INVOKE_URL = "https://mast.stsci.edu/api/v0/invoke"


def _mast_invoke(payload: dict[str, Any], timeout_sec: float = 30.0) -> dict[str, Any]:
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


def get_galex_jpeg_url(
    ra_deg: float,
    dec_deg: float,
    radius_deg: float,
    galex_band: str = "NUV",
) -> str:
    band = galex_band.strip().upper()
    if band not in ("NUV", "FUV"):
        raise ValueError("galex_band must be 'NUV' or 'FUV'")

    request = {
        "service": "Mast.Caom.Filtered.Position",
        "format": "json",
        "params": {
            "columns": "*",
            "filters": [
                {"paramName": "obs_collection", "values": ["GALEX"]},
                {"paramName": "dataproduct_type", "values": ["image"]},
                {"paramName": "filters", "values": [band], "separator": ";"},
            ],
            "position": f"{ra_deg}, {dec_deg}, {radius_deg}",
        },
        "pagesize": 200,
        "page": 1,
        "removenullcolumns": True,
        "removecache": True,
    }

    out = _mast_invoke(request, timeout_sec=30.0)
    data = out.get("data") or []
    for row in data:
        jpeg_url = row.get("jpegURL")
        if isinstance(jpeg_url, str) and jpeg_url.strip():
            return jpeg_url.strip()

    raise ValueError("MAST GALEX query returned no jpegURL results for this position/band.")
