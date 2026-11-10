"""Viewer analysis without a canvas capture when hips2fits (CDS only) is down:
the same field is downloaded from an equivalent survey hosted elsewhere."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import requests

from packages.galaxy_agent.agent_tools import make_tool_resolve_and_fetch_image
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.models import AnalyzeRequest
from packages.galaxy_core.infrastructure import availability

HIPS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits?x"


class _Store:
    def __init__(self) -> None:
        self.saved: list[bytes] = []

    def save_image(self, request_id: str, data: bytes) -> Any:
        self.saved.append(data)
        return MagicMock(path=f"/tmp/{request_id}-{len(self.saved)}.jpg")


def _request(hips_id: str) -> AnalyzeRequest:
    return AnalyzeRequest(
        request_id="req-view",
        message="analyze",
        task="morphology_summary",
        view_ra_deg=202.48,
        view_dec_deg=47.23,
        view_size_arcmin=8.0,
        view_hips_id=hips_id,
    )


@pytest.fixture
def setup(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    state: dict[str, Any] = {"urls": [], "hips_down": True, "sdss_down": False}
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.load_image", lambda _: np.ones((50, 50), np.float32)
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.get_image_url_from_hips_id",
        lambda ra, dec, hips_id, size: HIPS_URL,
    )
    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.survey_provider_registry.sdss_get_image_url",
        lambda ra_deg, dec_deg, size_arcmin: "https://skyserver.sdss.org/cutout.jpg",
    )

    def fake_dss2(ra_deg: float, dec_deg: float, size_arcmin: float, plate: str) -> bytes:
        state["urls"].append(f"dss2:{plate}")
        return b"dss2-image"

    monkeypatch.setattr("packages.galaxy_agent.agent_tools.fetch_dss2_cutout_jpeg", fake_dss2)

    def fake_get(url: str, timeout: Any) -> MagicMock:
        state["urls"].append(url)
        down = (url == HIPS_URL and state["hips_down"]) or ("sdss" in url and state["sdss_down"])
        if down:
            raise requests.ConnectTimeout("down")
        return MagicMock(content=b"image", raise_for_status=lambda: None)

    monkeypatch.setattr("packages.galaxy_agent.agent_tools._http.get", fake_get)
    return state


def _run(hips_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = ContextRegistry("req-view")
    tool = make_tool_resolve_and_fetch_image(registry, _Store(), _request(hips_id))
    result = tool.invoke({})
    return result, registry.get("coordinates:req-view")  # type: ignore[return-value]


def test_hips2fits_still_used_when_it_works(setup: dict[str, Any]) -> None:
    setup["hips_down"] = False
    result, coords = _run("CDS/P/SDSS9/color")
    assert setup["urls"] == [HIPS_URL]
    assert result["survey_used"] == "CDS/P/SDSS9/color"
    assert "image_fallback_from" not in coords


def test_sdss_field_from_skyserver_when_hips2fits_is_down(setup: dict[str, Any]) -> None:
    result, coords = _run("CDS/P/SDSS9/color")
    assert setup["urls"] == [HIPS_URL, "https://skyserver.sdss.org/cutout.jpg"]
    assert result["survey_used"] == "SDSS"
    assert coords["image_fallback_from"] == "CDS/P/SDSS9/color"
    assert (coords["ra_deg"], coords["dec_deg"], coords["size_arcmin"]) == (202.48, 47.23, 8.0)


def test_dss2_when_both_hips2fits_and_sdss_fail(setup: dict[str, Any]) -> None:
    setup["sdss_down"] = True
    result, _ = _run("CDS/P/SDSS9/color")
    assert setup["urls"][-1] == "dss2:poss2ukstu_red"
    assert result["survey_used"] == "DSS2"


def test_short_viewer_ids_also_fall_back(setup: dict[str, Any]) -> None:
    result, _ = _run("DSS2")
    assert result["survey_used"] == "DSS2"


def test_hips2fits_is_skipped_during_the_cool_down(setup: dict[str, Any]) -> None:
    _run("CDS/P/SDSS9/color")
    assert not availability.is_available("hips2fits")
    setup["urls"].clear()
    _run("CDS/P/SDSS9/color")
    assert HIPS_URL not in setup["urls"]


def test_no_other_band_is_used_as_a_substitute(setup: dict[str, Any]) -> None:
    with pytest.raises(requests.ConnectTimeout):
        _run("CDS/P/allWISE/color")
    assert setup["urls"] == [HIPS_URL]
