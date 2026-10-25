"""Numerical and scientific-contract regressions, with no LLM or network calls."""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from packages.galaxy_agent.analysis_quality import build_analysis_quality
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.tools import load_image
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer
from packages.galaxy_core.application.analyses.radial_profile import RadialProfileAnalysis
from packages.galaxy_core.application.analyses.sersic import _plot_sersic_residual
from packages.galaxy_core.application.morphology import (
    compute_basic_measurements,
    compute_radial_profile,
    compute_statmorph_metrics,
)
from packages.galaxy_core.domain.analysis import AnalysisContext
from packages.galaxy_core.domain.errors import NoSourceDetectedError
from packages.galaxy_core.domain.models import SegmentationResult


def _morph(flag=0, value=1.0):
    fields = (
        "concentration asymmetry smoothness ellipticity_asymmetry elongation_asymmetry "
        "orientation_asymmetry xc_asymmetry yc_asymmetry rpetro_circ r20 r80 sn_per_pixel "
        "sersic_n sersic_rhalf sersic_ellip sersic_theta sersic_amplitude sersic_xc "
        "sersic_yc sersic_chi2_dof"
    ).split()
    return SimpleNamespace(
        **dict.fromkeys(fields, value),
        flag=flag,
        flag_sersic=0 if flag < 2 else 4,
        _slice_stamp=(slice(1, 7), slice(2, 8)),
        _sersic_model=lambda x, y: 3.5 + x + 2 * y,
    )


def test_sigma_and_neighbour_labels_are_passed_without_inversion(monkeypatch):
    seen = {}

    def fake(image, labels, label, **kwargs):
        seen.update(kwargs, labels=labels.copy(), label=label)
        return _morph()

    monkeypatch.setattr("statmorph.SourceMorphology", fake)
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:7, 2:7] = 7
    labels[1, 8] = 9
    result = compute_statmorph_metrics(np.ones((10, 10)), labels, 7, background_rms=2)
    assert np.all(seen["weightmap"] == 2)
    assert seen["labels"][1, 8] == 9
    assert seen["label"] == 7
    assert seen["psf"] is None
    yy, xx = np.indices((6, 6))
    np.testing.assert_allclose(result["_sersic_diagnostic"]["model"], 3.5 + xx + 2 * yy)


def test_explicit_neighbour_mask_is_preserved(monkeypatch):
    seen = {}

    def fake(*args, **kwargs):
        seen.update(kwargs)
        return _morph()

    monkeypatch.setattr("statmorph.SourceMorphology", fake)
    excluded = np.zeros((10, 10), dtype=bool)
    excluded[0, 0] = True
    compute_statmorph_metrics(
        np.ones((10, 10)),
        np.ones((10, 10), dtype=int),
        1,
        noise_sigma=np.full((10, 10), 3.0),
        excluded_mask=excluded,
    )
    assert seen["mask"][0, 0]
    assert seen["weightmap"][0, 0] == 0
    assert seen["weightmap"][1, 1] == 3


@pytest.mark.parametrize("flag", [2, 4])
def test_invalid_metrics_are_not_published_as_finite_sentinels(monkeypatch, flag):
    monkeypatch.setattr("statmorph.SourceMorphology", lambda *a, **kw: _morph(flag, -99.0))
    result = compute_statmorph_metrics(np.ones((10, 10)), np.ones((10, 10), int), 1, 2)
    assert all(
        result[key] is None for key in ("concentration", "asymmetry", "smoothness", "sersic_n")
    )
    assert "_sersic_diagnostic" not in result


def test_small_negative_corrected_asymmetry_is_not_clipped(monkeypatch):
    monkeypatch.setattr("statmorph.SourceMorphology", lambda *a, **kw: _morph(0, -0.03))
    result = compute_statmorph_metrics(np.ones((10, 10)), np.ones((10, 10), int), 1, 2)
    assert result["asymmetry"] == -0.03


@pytest.mark.parametrize("value", [0.0, 7.0])
def test_blank_field_never_becomes_a_quantile_galaxy(value):
    with pytest.raises(NoSourceDetectedError):
        BasicGalaxyAnalyzer().segment_galaxy(np.full((64, 64), value, dtype=np.float32))


def test_basic_moments_measure_real_nonzero_ellipticity():
    yy, xx = np.indices((128, 128))
    data = np.exp(-0.5 * (((xx - 63.2) / 12) ** 2 + ((yy - 62.7) / 6) ** 2))
    mask = (data > 1e-5).astype(np.uint8)
    seg = SegmentationResult(mask, {}, data_sub=data)
    metrics = compute_basic_measurements(data, seg)
    assert metrics["centroid_x"] == pytest.approx(63.2, abs=0.01)
    assert metrics["centroid_y"] == pytest.approx(62.7, abs=0.01)
    assert metrics["ellipticity"] == pytest.approx(0.5, abs=0.01)


def test_radial_profile_uses_subtracted_data_and_masks_contaminants():
    yy, xx = np.indices((64, 64))
    labels = np.where(np.hypot(xx - 32, yy - 32) < 20, 1, 0).astype(int)
    labels[30:33, 40:44] = 2
    subtracted = np.full((64, 64), 4.0)
    subtracted[labels == 2] = 10000
    sigma = np.full((64, 64), 2.0)
    seg = SegmentationResult(
        (labels == 1).astype(np.uint8),
        {},
        data_sub=subtracted,
        tight_segmap_labels=labels,
        noise_sigma=sigma,
    )
    result = RadialProfileAnalysis().run(AnalysisContext(subtracted + 100, seg))
    profile = result.metrics["radial_profile"]
    np.testing.assert_allclose(profile["mean_flux"], 4.0)
    np.testing.assert_allclose(
        profile["mean_flux_err"],
        2 / np.sqrt(profile["n_pixels"]),
    )


def test_residual_plot_uses_exact_fitted_model_and_shared_intensity_scale(monkeypatch):
    import matplotlib.axes

    original = matplotlib.axes.Axes.imshow
    calls = []

    def record(self, values, *args, **kwargs):
        calls.append((np.asarray(values).copy(), kwargs.get("norm")))
        return original(self, values, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "imshow", record)
    data, model = np.full((10, 10), 9.0), np.full((10, 10), 3.0)
    output = _plot_sersic_residual(
        {
            "data": data,
            "model": model,
            "sigma": np.full((10, 10), 2.0),
            "mask": np.zeros((10, 10), bool),
        }
    )
    assert output.startswith(b"\x89PNG")
    np.testing.assert_allclose(calls[1][0], model)
    np.testing.assert_allclose(calls[2][0], data - model)
    np.testing.assert_allclose(calls[3][0], (data - model) / 2)
    assert calls[0][1] is calls[1][1]


def test_missing_modules_are_not_claimed_as_complete():
    quality = build_analysis_quality({"modules": {"cas": {"status": "ok"}}}, "morphology_summary")
    assert quality["status"] == "partial"
    assert quality["modules"]["sersic"] == "missing"
    assert quality["calibrated"] is False
    assert build_analysis_quality({}, "sersic")["status"] == "unavailable"


def test_original_artifact_is_immutable(tmp_path):
    store = ArtifactStore(str(tmp_path))
    artifact = store.save_image("test", b"original input")
    store.save_image("test", b"original input")
    with pytest.raises(ValueError, match="immutable"):
        store.save_image("test", b"changed input")
    assert (tmp_path / "test" / "image.jpg").read_bytes() == b"original input"
    assert artifact.type == "image"


def test_no_implicit_synthetic_image():
    with pytest.raises(ValueError, match="actual input"):
        load_image(None)


def test_grayscale_precision_is_not_reduced_to_eight_bits(tmp_path):
    path = tmp_path / "linear.tif"
    Image.fromarray(np.array([[10, 1000], [5000, 60000]], dtype=np.uint16)).save(path)
    np.testing.assert_array_equal(load_image(str(path)), [[10, 1000], [5000, 60000]])


def test_invalid_radial_bin_count_is_rejected():
    with pytest.raises(ValueError, match="n_bins"):
        compute_radial_profile(np.ones((10, 10)), np.ones((10, 10)), 5, 5, n_bins=-1)


@pytest.mark.parametrize("index", [1.0, 2.5])
def test_real_psf_convolved_sersic_recovers_known_parameters(index):
    from astropy.modeling.models import Sersic2D
    from scipy.signal import fftconvolve

    yy, xx = np.indices((128, 128), dtype=float)
    source = Sersic2D(amplitude=30, r_eff=10, n=index, x_0=63.2, y_0=64.3, ellip=0.3, theta=0.25)
    py, px = np.indices((11, 11), dtype=float)
    psf = np.exp(-((px - 5) ** 2 + (py - 5) ** 2) / (2 * 1.1**2))
    psf /= psf.sum()
    truth = fftconvolve(source(xx, yy), psf, mode="same")
    data = truth + np.random.default_rng(281).normal(0, 0.3, truth.shape)
    labels = (truth > 0.9).astype(np.int32)
    metrics = compute_statmorph_metrics(
        data, labels, 1, noise_sigma=np.full(data.shape, 0.3), psf=psf
    )
    assert metrics["flag_sersic"] == 0
    assert metrics["sersic_psf_corrected"] is True
    assert metrics["sersic_n"] == pytest.approx(index, rel=0.05)
    assert metrics["sersic_rhalf"] == pytest.approx(10, rel=0.05)
    assert metrics["sersic_ellip"] == pytest.approx(0.3, abs=0.03)
    assert metrics["sersic_xc"] == pytest.approx(63.2, abs=0.15)
    assert metrics["sersic_yc"] == pytest.approx(64.3, abs=0.15)


def _captured_request():
    from packages.galaxy_agent.models import AnalyzeRequest, Target

    return AnalyzeRequest(
        request_id="capture",
        target=Target(name="M101"),
        task="cas",
        image_data="pixels",
        view_ra_deg=12,
        view_dec_deg=20,
        view_size_arcmin=30,
        view_hips_id="SDSS",
        view_metadata={
            "ra_deg": 12,
            "dec_deg": 20,
            "fov_x_deg": 0.5,
            "fov_y_deg": 0.25,
            "width_px": 128,
            "height_px": 64,
            "hips_id": "SDSS",
            "projection": "SIN",
        },
    )


def test_full_rectangular_capture_is_preserved_when_resolving_request():
    from packages.galaxy_agent.models import Target
    from packages.galaxy_agent.view_integrity import validate_view_capture

    req = _captured_request()
    validate_view_capture(req, (64, 128))
    assert req.to_resolved_request(Target(name="M101"), "cas").view_metadata == req.view_metadata


@pytest.mark.parametrize(
    "changes",
    [
        {"ra_deg": 15},
        {"dec_deg": 21},
        {"fov_x_deg": 0.3},
        {"hips_id": "DSS2"},
        {"width_px": 64},
        {"capture_required": True},
        {"fov_y_deg": float("nan")},
    ],
)
def test_stale_capture_cannot_be_silently_analyzed(changes):
    from packages.galaxy_agent.view_integrity import validate_view_capture
    from packages.galaxy_core.domain.errors import ImageIntegrityError

    req = _captured_request()
    req.view_metadata.update(changes)
    with pytest.raises(ImageIntegrityError):
        validate_view_capture(req, (64, 128))


def test_isophote_fit_keeps_negative_noise_and_excludes_neighbours(monkeypatch):
    from packages.galaxy_core.application.isophotes import _fit_ellipses

    seen = []

    class FakeEllipse:
        def __init__(self, image, **kwargs):
            seen.append(image)

        def fit_image(self, **kwargs):
            return [
                SimpleNamespace(
                    valid=True,
                    sma=i,
                    intens=4,
                    eps=0.2,
                    pa=0.1,
                    x0=5,
                    y0=5,
                    int_err=0.3,
                    ellip_err=0.02,
                    pa_err=0.01,
                    stop_code=0,
                )
                for i in (4, 6, 8)
            ]

    monkeypatch.setattr("photutils.isophote.Ellipse", FakeEllipse)
    image = np.full((12, 12), -0.5)
    excluded = np.zeros(image.shape, bool)
    excluded[1, 1] = True
    table = _fit_ellipses(image, 5, 5, 0.2, 0, 4, 9, excluded)
    assert seen[0][2, 2] == -0.5
    assert seen[0].mask[1, 1]
    assert table[0]["intens_err"] == 0.3
    assert table[0]["stop_code"] == 0


def test_real_tool_pipeline_preserves_input_and_reports_module_quality(tmp_path):
    import base64
    import hashlib
    import io
    import json

    from packages.galaxy_agent.agent_tools import (
        make_tool_analyze,
        make_tool_generate_final_report,
        make_tool_resolve_and_fetch_image,
        make_tool_run_isophotes,
        make_tool_segment_image,
    )
    from packages.galaxy_agent.context_registry import ContextRegistry
    from packages.galaxy_agent.models import AnalyzeRequest, Target

    yy, xx = np.indices((128, 128))
    data = 15 + 130 * np.exp(-0.5 * (((xx - 64) / 13) ** 2 + ((yy - 64) / 9) ** 2))
    data += np.random.default_rng(8).normal(0, 1.5, data.shape)
    buffer = io.BytesIO()
    Image.fromarray(np.clip(data, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    original = buffer.getvalue()
    req = AnalyzeRequest(
        request_id="full-pipeline",
        target=Target(name="synthetic"),
        task="morphology_summary",
        view_ra_deg=12,
        view_dec_deg=20,
        view_size_arcmin=30,
        view_hips_id="DSS2",
        image_data=base64.b64encode(original).decode(),
    )
    store, registry, analyzer = (
        ArtifactStore(str(tmp_path)),
        ContextRegistry(req.request_id),
        BasicGalaxyAnalyzer(),
    )
    image = make_tool_resolve_and_fetch_image(registry, store, req).invoke({})
    segmented = make_tool_segment_image(registry, analyzer, store, req).invoke(
        {"image_handle": image["image_handle"]}
    )
    handles = {"image_handle": image["image_handle"], "seg_handle": segmented["seg_handle"]}
    make_tool_analyze(registry, analyzer, store, req).invoke(handles)
    make_tool_run_isophotes(registry, analyzer, store, req).invoke(handles)
    report = make_tool_generate_final_report(registry, None, req).invoke({})
    assert "exploratorio" in report["summary"]
    saved = tmp_path / req.request_id
    assert (saved / "image.jpg").read_bytes() == original
    payload = json.loads((saved / "measurements.json").read_text())
    assert payload["image_provenance"]["sha256"] == hashlib.sha256(original).hexdigest()
    assert all(name in payload["modules"] for name in ("cas", "sersic", "radial_profile"))
    assert (saved / "plot-segmentation.png").is_file()
