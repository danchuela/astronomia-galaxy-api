"""Tests for image resolution and download helpers in agent_tools.

These tests verify the band/catalog routing logic (_build_fetch_attempts),
target resolution (_resolve_target), and per-survey download routing
(_download_image). All external calls are monkeypatched.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from packages.galaxy_agent.agent_tools import (
    _build_fetch_attempts,
    _download_image,
    _resolve_target,
)
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.models import AnalyzeRequest, Artifact, Target


class _InMemoryArtifactStore(ArtifactStore):
    def __init__(self) -> None:
        super().__init__(base_dir="artifacts-test")
        self.saved_images: dict[str, bytes] = {}

    def save_image(self, request_id: str, image_bytes: bytes) -> Artifact:
        self.saved_images[request_id] = image_bytes
        return Artifact(
            type="image",
            path=f"memory://{request_id}/image.jpg",
        )


def _make_request(
    request_id: str = "req-1",
    target_name: str | None = "M81",
    options: dict[str, Any] | None = None,
) -> AnalyzeRequest:
    return AnalyzeRequest(
        request_id=request_id,
        message=None,
        messages=None,
        target=Target(name=target_name) if target_name is not None else None,
        task="measure_basic",
        image_url=None,
        options=options or {},
    )


def _fake_jpeg_bytes() -> bytes:
    return bytes(np.zeros((2, 2), dtype=np.uint8).tobytes())


# ---------------------------------------------------------------------------
# _build_fetch_attempts — band/catalog routing logic
# ---------------------------------------------------------------------------


def test_visible_band_prefers_sdss_first() -> None:
    attempts = _build_fetch_attempts({"band": "visible"})
    assert attempts == [("SDSS", None)]


def test_optical_band_maps_to_sdss() -> None:
    attempts = _build_fetch_attempts({"band": "optical"})
    assert attempts == [("SDSS", None)]


def test_infrared_band_forwards_band_param() -> None:
    attempts = _build_fetch_attempts({"band": "infrared"})
    assert attempts == [(None, "infrared")]


def test_explicit_catalog_overrides_band() -> None:
    attempts = _build_fetch_attempts({"catalog": "DSS2", "band": "infrared"})
    assert attempts == [("DSS2", None)]


def test_no_band_defaults_to_sdss() -> None:
    attempts = _build_fetch_attempts({})
    assert attempts == [("SDSS", None)]


# ---------------------------------------------------------------------------
# _resolve_target — failure propagation
# ---------------------------------------------------------------------------


def test_resolve_failure_raises_runtime_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.resolve_and_fetch",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("Simulated SDSS failure")),
    )
    req = _make_request(target_name="M81", options={"band": "visible"})
    with pytest.raises(RuntimeError, match="Failed to resolve"):
        _resolve_target(req, req.options or {})


# ---------------------------------------------------------------------------
# _download_image — per-survey client routing
# ---------------------------------------------------------------------------


def test_2mass_survey_routes_to_cutout_client(monkeypatch: pytest.MonkeyPatch) -> None:
    cutout_called = {"value": False}

    def fake_fetch_2mass(**kwargs: Any) -> bytes:
        cutout_called["value"] = True
        return _fake_jpeg_bytes()

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.fetch_2mass_cutout_jpeg",
        fake_fetch_2mass,
    )

    class _Resolved:
        ra_deg = 148.888
        dec_deg = 69.065
        survey_used = "2MASS-J"
        size_arcmin = 10.0
        image_url = "https://example.org/2mass.png"

    store = _InMemoryArtifactStore()
    path = _download_image("req-2mass", _Resolved(), store)  # type: ignore[arg-type]

    assert cutout_called["value"] is True
    assert path == "memory://req-2mass/image.jpg"
    assert store.saved_images["req-2mass"]


def test_dss2_survey_routes_to_dss2_client(monkeypatch: pytest.MonkeyPatch) -> None:
    cutout_called = {"value": False}

    def fake_fetch_dss2(**kwargs: Any) -> bytes:
        cutout_called["value"] = True
        return _fake_jpeg_bytes()

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.dss2_survey_to_plate",
        lambda _: "dss2",
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.fetch_dss2_cutout_jpeg",
        fake_fetch_dss2,
    )

    class _Resolved:
        ra_deg = 10.684
        dec_deg = 41.269
        survey_used = "DSS2"
        size_arcmin = 15.0
        image_url = "https://example.org/dss2.gif"

    store = _InMemoryArtifactStore()
    path = _download_image("req-dss2", _Resolved(), store)  # type: ignore[arg-type]

    assert cutout_called["value"] is True
    assert path == "memory://req-dss2/image.jpg"
