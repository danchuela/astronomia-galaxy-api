"""Unit tests for agent_tools factory functions.

Each tool is tested in isolation: factory called with a fake registry and
all external calls monkeypatched. Assertions focus on:
  - The handle stored in the registry after each tool runs
  - The JSON structure returned to the LLM
  - KeyError when a prerequisite handle is missing
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from packages.galaxy_agent.agent_tools import (
    make_tool_analyze,
    make_tool_enrich_metadata,
    make_tool_generate_final_report,
    make_tool_resolve_and_fetch_image,
    make_tool_segment_image,
)
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.models import AnalyzeRequest, Target
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _make_request(
    request_id: str = "req-test",
    task: str = "morphology_summary",
    target_name: str = "M31",
    options: dict[str, Any] | None = None,
) -> AnalyzeRequest:
    return AnalyzeRequest(
        request_id=request_id,
        message="analyze",
        target=Target(name=target_name),
        task=task,
        options=options or {},
    )


def _fake_image_bytes() -> bytes:
    return np.zeros((10, 10), dtype=np.uint8).tobytes()


def _fake_ndarray() -> np.ndarray:
    return np.ones((50, 50), dtype=np.float32) * 100.0


class _FakeArtifactStore:
    def __init__(self) -> None:
        self.saved: list[str] = []

    def save_image(self, request_id: str, image_bytes: bytes) -> Any:
        self.saved.append("image")
        m = MagicMock()
        m.path = f"artifacts/{request_id}/image.jpg"
        return m

    def save_mask(self, request_id: str, mask: Any) -> Any:
        self.saved.append("mask")
        m = MagicMock()
        m.type = "mask"
        m.path = f"artifacts/{request_id}/mask.png"
        return m

    def save_measurements(self, request_id: str, payload: Any) -> Any:
        return MagicMock()

    def save_plot(self, request_id: str, name: str, png_bytes: bytes) -> Any:
        m = MagicMock()
        m.type = "plot"
        m.path = f"artifacts/{request_id}/plot-{name}.png"
        return m


# ---------------------------------------------------------------------------
# resolve_and_fetch_image
# ---------------------------------------------------------------------------


def test_resolve_and_fetch_image_full_pipeline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Full resolve path stores image array in registry and returns expected keys."""
    fake_image = _fake_ndarray()
    fake_resolved = MagicMock()
    fake_resolved.ra_deg = 10.68
    fake_resolved.dec_deg = 41.27
    fake_resolved.survey_used = "SDSS"
    fake_resolved.size_arcmin = 10.0
    fake_resolved.image_url = "https://example.org/image.jpg"

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.resolve_and_fetch", lambda **_: fake_resolved
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools._http.get",
        lambda url, timeout: MagicMock(content=_fake_image_bytes(), raise_for_status=lambda: None),
    )
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.dss2_survey_to_plate", lambda _: None)
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.load_image", lambda _: fake_image)

    registry = ContextRegistry("req-001")
    store = _FakeArtifactStore()
    request = _make_request("req-001")

    tool = make_tool_resolve_and_fetch_image(registry, store, request)
    result = tool.invoke({})

    assert result["image_handle"] == "image:req-001"
    assert abs(result["ra_deg"] - 10.68) < 0.01
    assert result["survey_used"] == "SDSS"
    stored = registry.get("image:req-001")
    assert stored is fake_image


def test_resolve_and_fetch_image_viewer_base64(monkeypatch: pytest.MonkeyPatch) -> None:
    """Viewer canvas path uses the exact framed image from the frontend."""
    import base64

    fake_image = _fake_ndarray()
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.load_image", lambda _: fake_image)

    registry = ContextRegistry("req-canvas")
    store = _FakeArtifactStore()
    fake_bytes = _fake_image_bytes()
    b64 = "data:image/jpeg;base64," + base64.b64encode(fake_bytes).decode()
    request = AnalyzeRequest(
        request_id="req-canvas",
        message="analyze",
        task="morphology_summary",
        view_ra_deg=10.68,
        view_dec_deg=41.27,
        view_hips_id="CDS/P/SDSS9/color",
        image_data=b64,
    )

    tool = make_tool_resolve_and_fetch_image(registry, store, request)
    result = tool.invoke({})

    assert result["image_handle"] == "image:req-canvas"
    assert result["survey_used"] == "CDS/P/SDSS9/color"
    assert registry.get("image:req-canvas") is fake_image


def test_resolve_and_fetch_image_prefers_viewer_canvas_over_hips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Viewer analysis preserves the exact canvas capture when both sources exist."""
    import base64

    fake_image = _fake_ndarray()
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.load_image", lambda _: fake_image)
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.get_image_url_from_hips_id",
        lambda ra, dec, hips_id, size: "https://example.org/hips2fits.jpg",
    )

    def fake_get(url: str, timeout: int) -> MagicMock:
        raise AssertionError("HiPS should not be used when canvas image_data is present")

    monkeypatch.setattr("packages.galaxy_agent.agent_tools._http.get", fake_get)

    registry = ContextRegistry("req-hips")
    store = _FakeArtifactStore()
    b64 = "data:image/jpeg;base64," + base64.b64encode(_fake_image_bytes()).decode()
    request = AnalyzeRequest(
        request_id="req-hips",
        message="analyze",
        task="morphology_summary",
        view_ra_deg=10.68,
        view_dec_deg=41.27,
        view_hips_id="CDS/P/SDSS9/color",
        image_data=b64,
    )

    tool = make_tool_resolve_and_fetch_image(registry, store, request)
    result = tool.invoke({})

    assert result["image_handle"] == "image:req-hips"
    assert result["survey_used"] == "CDS/P/SDSS9/color"
    assert registry.get("image:req-hips") is fake_image


def test_resolve_and_fetch_image_uses_hips_when_canvas_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """HiPS remains the fallback when the viewer has no canvas capture."""
    fake_image = _fake_ndarray()
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.load_image", lambda _: fake_image)
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.get_image_url_from_hips_id",
        lambda ra, dec, hips_id, size: "https://example.org/hips2fits.jpg",
    )

    requested_urls: list[str] = []

    def fake_get(url: str, timeout: int) -> MagicMock:
        requested_urls.append(url)
        return MagicMock(content=b"hips-image", raise_for_status=lambda: None)

    monkeypatch.setattr("packages.galaxy_agent.agent_tools._http.get", fake_get)

    registry = ContextRegistry("req-hips-fallback")
    store = _FakeArtifactStore()
    request = AnalyzeRequest(
        request_id="req-hips-fallback",
        message="analyze",
        task="morphology_summary",
        view_ra_deg=10.68,
        view_dec_deg=41.27,
        view_hips_id="CDS/P/SDSS9/color",
    )

    tool = make_tool_resolve_and_fetch_image(registry, store, request)
    result = tool.invoke({})

    assert result["image_handle"] == "image:req-hips-fallback"
    assert result["survey_used"] == "CDS/P/SDSS9/color"
    assert requested_urls == ["https://example.org/hips2fits.jpg"]
    assert registry.get("image:req-hips-fallback") is fake_image


# ---------------------------------------------------------------------------
# segment_image
# ---------------------------------------------------------------------------


def test_segment_image_stores_seg_handle(monkeypatch: pytest.MonkeyPatch) -> None:
    """segment_image reads image handle, calls tool_segment, stores SegmentationResult."""
    fake_image = _fake_ndarray()
    fake_seg = MagicMock()
    fake_seg.mask = np.ones((50, 50), dtype=np.uint8)
    fake_seg.metadata = {}

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.tool_segment",
        lambda analyzer, img, thresh_sigma: fake_seg,
    )

    registry = ContextRegistry("req-seg")
    registry.put("image:req-seg", fake_image)
    store = _FakeArtifactStore()
    request = _make_request("req-seg")
    analyzer = BasicGalaxyAnalyzer()

    tool = make_tool_segment_image(registry, analyzer, store, request)
    result = tool.invoke({"image_handle": "image:req-seg"})

    assert result["seg_handle"] == "seg:req-seg"
    assert "mask_ratio" in result
    assert registry.get("seg:req-seg") is fake_seg


def test_segment_image_raises_if_image_handle_missing() -> None:
    """segment_image raises KeyError when image was not yet fetched."""
    registry = ContextRegistry("req-missing")
    store = _FakeArtifactStore()
    request = _make_request("req-missing")
    analyzer = BasicGalaxyAnalyzer()

    tool = make_tool_segment_image(registry, analyzer, store, request)
    with pytest.raises(KeyError):
        tool.invoke({"image_handle": "image:req-missing"})


# ---------------------------------------------------------------------------
# analyze_galaxy
# ---------------------------------------------------------------------------


def test_analyze_galaxy_returns_metrics_and_morphology_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """analyze_galaxy reads both handles, runs analysis, stores metrics in registry."""
    fake_image = _fake_ndarray()
    fake_seg = MagicMock()
    fake_seg.mask = np.zeros((50, 50), dtype=np.uint8)
    fake_seg.mask[10:40, 10:40] = 1
    fake_seg.metadata = {}

    fake_ar = MagicMock()
    fake_ar.metrics = {"ellipticity": 0.3}
    fake_ar.summary = "Galaxy is elliptical."
    fake_ar.image_png = None
    fake_ar.module_name = "test"

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.tool_run_analysis", lambda *args, **kwargs: [fake_ar]
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools._annotate_image_in_place", lambda *a: None
    )

    registry = ContextRegistry("req-analyze")
    registry.put("image:req-analyze", fake_image)
    registry.put("seg:req-analyze", fake_seg)
    store = _FakeArtifactStore()
    request = _make_request("req-analyze", task="morphology_summary")
    analyzer = BasicGalaxyAnalyzer()

    tool = make_tool_analyze(registry, analyzer, store, request)
    result = tool.invoke({"image_handle": "image:req-analyze", "seg_handle": "seg:req-analyze"})

    assert "metrics" in result
    assert "morphology_text" in result
    assert "Galaxy is elliptical." in result["morphology_text"]
    stored = registry.get("metrics:req-analyze")
    assert stored is not None


def test_analyze_galaxy_raises_if_seg_handle_missing() -> None:
    """analyze_galaxy raises KeyError when segment_image was not called first."""
    registry = ContextRegistry("req-noseg")
    registry.put("image:req-noseg", _fake_ndarray())
    store = _FakeArtifactStore()
    request = _make_request("req-noseg")
    analyzer = BasicGalaxyAnalyzer()

    tool = make_tool_analyze(registry, analyzer, store, request)
    with pytest.raises(KeyError):
        tool.invoke({"image_handle": "image:req-noseg", "seg_handle": "seg:req-noseg"})


# ---------------------------------------------------------------------------
# enrich_metadata
# ---------------------------------------------------------------------------


def test_enrich_metadata_returns_simbad_and_hst(monkeypatch: pytest.MonkeyPatch) -> None:
    """enrich_metadata calls SIMBAD and MAST and returns formatted strings."""
    fake_simbad = {"morph_type": "Sb", "object_type": "G"}
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.simbad_query_object", lambda name: fake_simbad
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.format_object_info", lambda r: "SIMBAD info text"
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.search_hst_jwst", lambda ra, dec: {"count": 5}
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.format_hst_jwst_info", lambda r: "HST info text"
    )

    registry = ContextRegistry("req-enrich")
    registry.put("coordinates:req-enrich", {"ra_deg": 10.68, "dec_deg": 41.27})
    request = _make_request("req-enrich")

    tool = make_tool_enrich_metadata(registry, request)
    result = tool.invoke({})

    assert result["object_info"] == "SIMBAD info text"
    assert result["hst_jwst"] == "HST info text"
    assert registry.get("object_info:req-enrich") == fake_simbad


def test_enrich_metadata_handles_simbad_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    """enrich_metadata silently handles SIMBAD errors and still returns empty strings."""
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.simbad_query_object",
        lambda _: (_ for _ in ()).throw(RuntimeError("timeout")),
    )
    monkeypatch.setattr("packages.galaxy_agent.agent_tools.search_hst_jwst", lambda ra, dec: None)

    registry = ContextRegistry("req-simbad-fail")
    registry.put("coordinates:req-simbad-fail", {"ra_deg": 10.68, "dec_deg": 41.27})
    request = _make_request("req-simbad-fail")

    tool = make_tool_enrich_metadata(registry, request)
    result = tool.invoke({})

    assert result["object_info"] == ""
    assert result["hst_jwst"] == ""


# ---------------------------------------------------------------------------
# generate_final_report
# ---------------------------------------------------------------------------


def test_generate_final_report_calls_langchain_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """generate_final_report calls generate_accompanying_summary when backend is present."""
    fake_backend = MagicMock()
    fake_backend.generate_accompanying_summary.return_value = "Final LLM summary."

    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools.append_catalog_and_field",
        lambda summary, catalog, size_arcmin: summary,
    )

    registry = ContextRegistry("req-report")
    request = _make_request("req-report")

    tool = make_tool_generate_final_report(registry, fake_backend, request)
    result = tool.invoke(
        {"morphology_text": "Raw morphology text.", "object_info": "", "hst_jwst_info": ""}
    )

    assert result["summary"] == "Final LLM summary."
    fake_backend.generate_accompanying_summary.assert_called_once()
    assert registry.get("summary:req-report") == "Final LLM summary."


def test_generate_final_report_no_backend() -> None:
    """generate_final_report returns morphology_text as-is when no backend."""
    registry = ContextRegistry("req-nobackend")
    request = _make_request("req-nobackend")

    tool = make_tool_generate_final_report(registry, None, request)
    result = tool.invoke({"morphology_text": "Plain text.", "object_info": "", "hst_jwst_info": ""})

    assert "Plain text." in result["summary"]
