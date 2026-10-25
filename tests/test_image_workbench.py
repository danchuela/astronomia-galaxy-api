"""Selection, ancillary planes, stability, inspection and capacity contracts."""

import asyncio
import base64
import io

import numpy as np
import pytest
from astropy.io import fits
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

from apps.api.capacity import AnalysisCapacity
from apps.api.main import app
from packages.galaxy_agent.image_inspection import inspect_image, segmentation_inputs
from packages.galaxy_agent.image_selection import ImageRegion, SourcePoint
from packages.galaxy_agent.uploads import UploadedImage, UploadedImageError, prepare_upload
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer
from packages.galaxy_core.application.morphology import compute_basic_measurements
from packages.galaxy_core.application.uncertainty import estimate_noise_stability
from packages.galaxy_core.domain.errors import ImageCapacityError, NoSourceDetectedError


def field():
    yy, xx = np.indices((128, 128))
    return (
        100
        + 400 * np.exp(-((xx - 63) ** 2 + (yy - 64) ** 2) / 90)
        + 300 * np.exp(-((xx - 24) ** 2 + (yy - 25) ** 2) / 45)
        + np.random.default_rng(21).normal(0, 1, (128, 128))
    )


def upload(**options):
    yy, xx = np.indices((17, 17))
    psf = np.exp(-((xx - 8) ** 2 + (yy - 8) ** 2) / 4)
    hdus = [
        fits.PrimaryHDU(),
        fits.ImageHDU(field(), name="SCI"),
        fits.ImageHDU(np.full((128, 128), 4.0), name="ERR"),
        fits.ImageHDU(psf, name="PSF"),
    ]
    buffer = io.BytesIO()
    fits.HDUList(hdus).writeto(buffer)
    return UploadedImage(
        filename="field.fits", data=base64.b64encode(buffer.getvalue()).decode(), **options
    )


def test_manual_source_overrides_central_source_and_keeps_original_pixels():
    item = upload(source_point={"x": 24 / 127, "y": 25 / 127})
    prepared = prepare_upload(item)
    seg = BasicGalaxyAnalyzer().segment_galaxy(
        prepared.pixels, **segmentation_inputs(item, prepared)
    )
    metrics = compute_basic_measurements(prepared.pixels, seg)
    assert metrics["centroid_x"] == pytest.approx(24, abs=0.5)
    assert metrics["centroid_y"] == pytest.approx(25, abs=0.5)
    assert seg.metadata["selection_method"] == "user_point"
    assert prepared.pixels.shape == (128, 128)
    np.testing.assert_array_equal(prepared.pixels, field())


def test_background_click_does_not_silently_select_another_source():
    with pytest.raises(NoSourceDetectedError, match="selected point"):
        BasicGalaxyAnalyzer().segment_galaxy(field(), source_point=(0, 0))


def test_region_excludes_every_pixel_outside_and_marks_clipped_source():
    seg = BasicGalaxyAnalyzer().segment_galaxy(
        field(), source_point=(24, 25), region=(16, 17, 31, 34)
    )
    assert seg.excluded_mask[64, 63]
    assert seg.metadata["region_applied"] == 1 and seg.metadata["touches_boundary"] == 1
    assert not seg.mask[64, 63]


@pytest.mark.parametrize("point", [{"x": -1, "y": 0.5}, {"x": 0.5, "y": float("nan")}])
def test_invalid_points_are_rejected(point):
    with pytest.raises(ValidationError):
        SourcePoint(**point)


def test_reversed_or_empty_regions_are_rejected():
    with pytest.raises(ValidationError):
        ImageRegion(left=0.9, right=0.2, top=0, bottom=1)


@pytest.mark.parametrize("kind,sigma", [("sigma", 4), ("variance", 2), ("inverse_variance", 0.5)])
def test_error_planes_are_explicitly_converted_to_sigma(kind, sigma):
    prepared = prepare_upload(upload(error_hdu_index=2, error_kind=kind, psf_hdu_index=3))
    np.testing.assert_array_equal(prepared.noise_sigma, sigma)
    assert prepared.psf.sum() == pytest.approx(1)
    assert prepared.provenance["noise_model"] == "user_sigma"
    assert prepared.provenance["psf"] == "user_supplied"


@pytest.mark.parametrize(
    "options",
    [
        {"psf_hdu_index": 1},
        {"error_hdu_index": 1},
        {"psf_hdu_index": 2},
        {"error_hdu_index": 3},
        {"psf_hdu_index": 3, "error_hdu_index": 3},
    ],
)
def test_auxiliary_planes_cannot_be_science_wrong_shape_or_shared(options):
    with pytest.raises(UploadedImageError, match="ancillary"):
        prepare_upload(upload(**options))


def test_inspection_has_normalized_preview_headers_and_detections_without_llm(monkeypatch):
    result = inspect_image(upload())
    assert result["width"] == result["height"] == 128
    assert result["preview"].startswith("data:image/png;base64,")
    assert len(result["sources"]) >= 2
    assert sum(item["automatic"] for item in result["sources"]) == 1
    assert [item["index"] for item in result["hdus"]] == [1, 2, 3]
    assert "data" not in result


def test_inspection_http_is_bounded_and_invalid_bytes_return_clarification():
    with TestClient(app) as client:
        assert client.post("/images/inspect", json=upload().model_dump()).status_code == 200
        assert (
            client.post("/images/inspect", json={"filename": "bad.png", "data": "AAAA"}).status_code
            == 422
        )
        assert (
            client.post("/images/inspect", content=b"x" * (12 * 1024 * 1024 + 1)).status_code == 413
        )


def test_noise_stability_is_deterministic_reports_failed_samples_and_excludes_systematics(
    monkeypatch,
):
    seg = BasicGalaxyAnalyzer().segment_galaxy(field())

    def metrics(data, *args, **kwargs):
        return {"concentration": float(np.mean(data))}

    monkeypatch.setattr(
        "packages.galaxy_core.application.uncertainty.compute_statmorph_metrics", metrics
    )
    first = estimate_noise_stability(seg)
    assert first == estimate_noise_stability(seg)
    assert first["intervals"]["concentration"]["std"] > 0
    assert first["intervals"]["concentration"]["valid_samples"] == 16
    assert "correlated_noise" in first["excludes"]
    assert "segmentation" in first["fixed"]


def test_unavailable_stability_never_invents_error_bars(monkeypatch):
    seg = BasicGalaxyAnalyzer().segment_galaxy(field())
    monkeypatch.setattr(
        "packages.galaxy_core.application.uncertainty.compute_statmorph_metrics",
        lambda *a, **kw: {},
    )
    assert estimate_noise_stability(seg)["status"] == "unavailable"


def test_capacity_rejects_excess_load_but_keeps_health_and_releases_slot():
    import httpx

    small = FastAPI()
    small.add_middleware(AnalysisCapacity)
    entered = asyncio.Event()
    release = asyncio.Event()

    @small.post("/images/inspect")
    async def inspect():
        entered.set()
        await release.wait()
        return {"ok": True}

    @small.get("/health")
    def health():
        return {"status": "ok"}

    async def check():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=small), base_url="http://test"
        ) as client:
            first = asyncio.create_task(client.post("/images/inspect"))
            await entered.wait()
            others = await asyncio.gather(*(client.post("/images/inspect") for _ in range(8)))
            assert all(
                item.status_code == 503 and item.headers["retry-after"] == "3" for item in others
            )
            assert (await client.get("/health")).status_code == 200
            release.set()
            assert (await first).status_code == 200
            assert (await client.post("/images/inspect")).status_code == 200

    asyncio.run(check())


def test_sep_pixel_budget_is_bounded_restored_and_not_reported_as_a_service_error(monkeypatch):
    import sep

    from packages.galaxy_core.application.morphology import compute_sep_segmap

    prior = sep.get_extract_pixstack()

    def full(*args, **kwargs):
        assert sep.get_extract_pixstack() == 300_000
        raise Exception("internal pixel buffer full")

    monkeypatch.setattr(sep, "extract", full)
    with pytest.raises(ImageCapacityError):
        compute_sep_segmap(np.ones((1025, 1025)), 1)
    assert sep.get_extract_pixstack() == prior


def test_large_morphology_stamp_is_rejected_before_statmorph_allocates(monkeypatch):
    from packages.galaxy_core.application.morphology import compute_statmorph_metrics

    monkeypatch.setattr(
        "statmorph.SourceMorphology", lambda *a, **kw: pytest.fail("Over-budget fit")
    )
    with pytest.raises(ImageCapacityError):
        compute_statmorph_metrics(np.ones((1024, 1024)), np.ones((1024, 1024), dtype=int), 1, 1)


def test_dense_source_is_rejected_before_sep_allocates_its_deblending_buffers(monkeypatch):
    from packages.galaxy_core.application.morphology import compute_sep_segmap

    monkeypatch.setattr("sep.extract", lambda *a, **kw: pytest.fail("Over-budget detection"))
    with pytest.raises(ImageCapacityError):
        compute_sep_segmap(np.full((600, 600), 5.0), 1)


def test_inspection_keeps_the_preview_available_when_detection_is_over_budget(monkeypatch):
    def limited(*a, **kw):
        raise ImageCapacityError("Detection budget")

    monkeypatch.setattr("packages.galaxy_agent.image_inspection.compute_sep_segmap", limited)
    result = inspect_image(upload())
    assert result["detection_status"] == "resource_limit" and result["sources"] == []
    assert result["preview"].startswith("data:image/png;base64,")
