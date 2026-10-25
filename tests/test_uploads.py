import base64
import hashlib
import io
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.io import fits
from fastapi.testclient import TestClient
from PIL import Image
from pydantic import ValidationError

from apps.api.main import app
from packages.galaxy_agent.agent_runner import AgentRunner
from packages.galaxy_agent.agent_tools import (
    make_tool_analyze,
    make_tool_enrich_metadata,
    make_tool_generate_final_report,
    make_tool_resolve_and_fetch_image,
    make_tool_segment_image,
)
from packages.galaxy_agent.analysis_quality import build_analysis_quality
from packages.galaxy_agent.artifacts import ArtifactStore
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.models import AnalyzeRequest
from packages.galaxy_agent.uploads import UploadedImage, UploadedImageError, prepare_upload
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer


def upload(raw, name="source.fits", **kwargs):
    return UploadedImage(filename=name, data=base64.b64encode(raw).decode(), **kwargs)


def fits_bytes(data):
    buffer = io.BytesIO()
    fits.PrimaryHDU(data).writeto(buffer)
    return buffer.getvalue()


@pytest.mark.parametrize("extension,format", [("png", "PNG"), ("jpg", "JPEG"), ("tif", "TIFF")])
def test_raster_upload_preserves_native_grayscale_pixels(extension, format):
    data = (
        np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
        if format != "JPEG"
        else np.full((32, 48), 30, dtype=np.uint8)
    )
    buffer = io.BytesIO()
    Image.fromarray(data).save(buffer, format=format)
    raw = buffer.getvalue()
    result = prepare_upload(upload(raw, "test." + extension))
    np.testing.assert_array_equal(result.pixels, data)
    assert result.original == raw
    assert result.provenance["sha256"] == hashlib.sha256(raw).hexdigest()
    assert Image.open(io.BytesIO(result.preview)).size == (48, 32)


def test_fits_keeps_negative_values_precision_and_nan_without_stretching():
    data = np.linspace(-23.4456789, 1806.99999, 1024).reshape(32, 32)
    data[0, 0] = np.nan
    prepared = prepare_upload(upload(fits_bytes(data)))
    np.testing.assert_array_equal(prepared.pixels, data)
    assert prepared.provenance["input_kind"] == "fits"
    assert prepared.provenance["calibrated"] is False
    assert Image.open(io.BytesIO(prepared.preview)).mode == "L"


def test_fits_bscale_bzero_are_applied_not_a_display_stretch():
    data = np.arange(1024, dtype=np.uint16).reshape(32, 32) + 32000
    prepared = prepare_upload(upload(fits_bytes(data)))
    np.testing.assert_array_equal(prepared.pixels, data)


def test_multi_hdu_requires_explicit_choice_but_single_sci_is_selected():
    buffer = io.BytesIO()
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((32, 32)), name="SCI"),
            fits.ImageHDU(np.ones((32, 32)) * 3, name="SCI"),
        ]
    ).writeto(buffer)
    with pytest.raises(UploadedImageError, match="planes"):
        prepare_upload(upload(buffer.getvalue()))
    np.testing.assert_array_equal(prepare_upload(upload(buffer.getvalue(), hdu_index=2)).pixels, 3)
    with pytest.raises(UploadedImageError, match="hdu"):
        prepare_upload(upload(buffer.getvalue(), hdu_index=0))
    buffer = io.BytesIO()
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(np.ones((32, 32)) * 7, name="SCI"),
            fits.ImageHDU(np.ones((32, 32)), name="ERR"),
        ]
    ).writeto(buffer)
    assert prepare_upload(upload(buffer.getvalue())).provenance["hdu_index"] == 1


@pytest.mark.parametrize("shape", [(10, 10), (2049, 2048), (3, 32, 32)])
def test_unusable_fits_dimensions_are_rejected_before_analysis(shape):
    with pytest.raises(UploadedImageError):
        prepare_upload(upload(fits_bytes(np.zeros(shape, dtype=np.uint8))))


@pytest.mark.parametrize(
    "raw,name",
    [(b"not an image", "fake.png"), (b"<svg></svg>", "fake.jpg"), (b"bad FITS", "bad.fits")],
)
def test_invalid_image_contents_are_not_accepted_by_extension(raw, name):
    with pytest.raises(UploadedImageError, match="invalid"):
        prepare_upload(upload(raw, name))


def test_bad_encoding_mime_spoof_and_transparency_rejected():
    with pytest.raises(UploadedImageError):
        prepare_upload(UploadedImage(filename="bad.png", data="notbase64!"))
    buffer = io.BytesIO()
    Image.new("RGBA", (32, 32), (255, 0, 0, 0)).save(buffer, "PNG")
    with pytest.raises(UploadedImageError):
        prepare_upload(upload(buffer.getvalue(), "alpha.png"))
    with pytest.raises(UploadedImageError):
        prepare_upload(upload(buffer.getvalue(), "spoof.jpg"))


def test_upload_schema_sanitizes_names_and_excludes_pixels_from_repr():
    image = UploadedImage(filename=r"C:\private\photo.png", data="sensitivepixels")
    assert image.filename == "photo.png" and "sensitivepixels" not in repr(image)
    with pytest.raises(ValidationError):
        UploadedImage(filename="image.svg", data="AAAA")
    with pytest.raises(ValidationError):
        UploadedImage(filename="image.png", data="a" * 11184913)


def test_file_is_a_valid_source_without_target_or_viewer_and_survives_resolution():
    req = AnalyzeRequest(request_id="upload", uploaded_image=upload(fits_bytes(np.ones((32, 32)))))
    runner = object.__new__(AgentRunner)
    req = runner._resolve_request(req)
    assert req.target is None and req.task == "morphology_summary"
    assert not runner._needs_target_prompt(req)
    assert not runner._must_open_viewer_before_analysis(req)
    assert req.to_resolved_request(None, "cas").uploaded_image == req.uploaded_image


def test_uploaded_image_pipeline_never_fetches_catalogue_or_analyzes_preview(tmp_path, monkeypatch):
    yy, xx = np.indices((128, 128))
    data = 20 + 300 * np.exp(-((xx - 64) ** 2 + (yy - 64) ** 2) / 180)
    data += np.random.default_rng(12).normal(0, 1, data.shape)
    raw = fits_bytes(data)
    req = AnalyzeRequest(
        request_id="pipeline",
        task="cas",
        uploaded_image=upload(raw),
        view_ra_deg=999,
        view_metadata={"capture_required": True},
    )
    registry, store, analyzer = (
        ContextRegistry(req.request_id),
        ArtifactStore(str(tmp_path)),
        BasicGalaxyAnalyzer(),
    )
    monkeypatch.setattr(
        "packages.galaxy_agent.agent_tools._resolve_target",
        lambda *a: pytest.fail("catalogue fetch"),
    )
    fetched = make_tool_resolve_and_fetch_image(registry, store, req).invoke({})
    np.testing.assert_array_equal(registry.get(fetched["image_handle"]), data)
    seg = make_tool_segment_image(registry, analyzer, store, req).invoke(
        {"image_handle": fetched["image_handle"]}
    )
    make_tool_analyze(registry, analyzer, store, req).invoke(
        {"image_handle": fetched["image_handle"], "seg_handle": seg["seg_handle"]}
    )
    assert make_tool_enrich_metadata(registry, req).invoke({}) == {
        "object_info": "",
        "hst_jwst": "",
    }
    report = make_tool_generate_final_report(registry, SimpleNamespace(), req).invoke({})["summary"]
    assert "source.fits" in report and "SDSS" not in report and "10.0" not in report
    assert (tmp_path / "pipeline" / "original.fits").read_bytes() == raw
    results = registry.get("results:pipeline")
    quality = build_analysis_quality(results, "cas")
    assert quality["input_kind"] == "fits" and "rendered_image" not in quality["issues"]
    assert quality["source"] == "upload"


def test_large_analysis_bodies_are_rejected_before_json_or_llm():
    with TestClient(app) as client:
        response = client.post("/analyze/stream", content=b"x" * (12 * 1024 * 1024 + 1))
    assert response.status_code == 413


def test_invalid_upload_is_a_clarification_before_creating_the_agent(tmp_path, monkeypatch):
    from packages.galaxy_agent.orchestrator import TaskOrchestrator

    monkeypatch.setattr(
        "packages.galaxy_agent.orchestrator.build_galaxy_agent",
        lambda *a: pytest.fail("No model needed for invalid bytes"),
    )
    request = AnalyzeRequest(
        request_id="invalid", task="cas", uploaded_image=upload(b"not an image", "bad.png")
    )
    orchestrator = TaskOrchestrator(BasicGalaxyAnalyzer(), ArtifactStore(str(tmp_path)))
    events = list(orchestrator.run_stream(request, False))
    assert events[-1]["analysis_quality"]["reason"] == "invalid_upload"
    assert events[-1]["status"] == "success" and events[-1]["artifacts"] == []
    assert "archivo" in events[-1]["summary"] and "problemas" not in events[-1]["summary"]


def test_language_parser_keeps_attachment_and_never_sends_its_bytes_to_model(monkeypatch):
    from unittest.mock import Mock

    from packages.galaxy_agent.langchain_backend import LangChainBackend

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    backend = LangChainBackend()
    backend._client = Mock()
    backend._client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"task":"cas","can_fulfill":true,"name":null}')
            )
        ]
    )
    request = AnalyzeRequest(
        request_id="parse",
        message="Dame CAS de esta imagen",
        uploaded_image=upload(fits_bytes(np.ones((32, 32)))),
        view_ra_deg=120,
        view_hips_id="DSS2",
        view_metadata={"capture_required": True},
    )
    result = backend.enrich_request(request)
    assert (
        result.uploaded_image == request.uploaded_image
        and result.target is None
        and result.task == "cas"
    )
    sent = backend._client.chat.completions.create.call_args.kwargs
    assert request.uploaded_image.data not in repr(sent)
    assert "IMAGEN ADJUNTA" in sent["messages"][0]["content"]
    assert "VISTA DEL VISOR ACTIVA" not in sent["messages"][0]["content"]


def test_only_supported_decoders_can_open_uploaded_bytes(monkeypatch):
    real_open = Image.open
    formats_seen = []

    def restricted_open(*args, **kwargs):
        formats_seen.append(kwargs.get("formats"))
        return real_open(*args, **kwargs)

    monkeypatch.setattr(Image, "open", restricted_open)
    with pytest.raises(UploadedImageError):
        prepare_upload(
            upload(b"#define x_width 32\n#define x_height 32\nstatic char x_bits[]={};", "fake.png")
        )
    assert formats_seen == [("PNG", "JPEG", "TIFF")]


def test_multipage_tiff_is_rejected_without_scanning_all_pages():
    buffer = io.BytesIO()
    image = Image.new("L", (32, 32), 30)
    image.save(buffer, format="TIFF", save_all=True, append_images=[image])
    with pytest.raises(UploadedImageError, match="planes"):
        prepare_upload(upload(buffer.getvalue(), "multiple.tiff"))
