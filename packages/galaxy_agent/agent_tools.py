"""LangChain tool factories for the galaxy analysis agent.

Each make_tool_* function returns a @tool-decorated closure that captures
the ContextRegistry and other dependencies. The registry stores non-serializable
objects (numpy arrays, SegmentationResult) keyed by opaque handle strings.
The LLM passes these handle strings between tool calls to chain the pipeline.
"""

from __future__ import annotations

import base64
import logging
from typing import Any

import numpy as np
import requests as _http
from langchain_core.tools import BaseTool, tool

from packages.galaxy_agent.artifacts import ArtifactWriter
from packages.galaxy_agent.context_registry import ContextRegistry
from packages.galaxy_agent.models import AnalyzeRequest, Artifact
from packages.galaxy_agent.request_utils import append_catalog_and_field, last_user_message
from packages.galaxy_agent.tools import (
    load_image,
    tool_isophotes,
    tool_run_analysis,
    tool_segment,
)
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer
from packages.galaxy_core.application.resolve_and_fetch_service import resolve_and_fetch
from packages.galaxy_core.domain import ResolvedTarget, SegmentationResult
from packages.galaxy_core.domain.imaging import BAND_TO_SURVEY
from packages.galaxy_core.infrastructure.dss2_client import (
    fetch_cutout_jpeg as fetch_dss2_cutout_jpeg,
)
from packages.galaxy_core.infrastructure.dss2_client import (
    survey_to_plate as dss2_survey_to_plate,
)
from packages.galaxy_core.infrastructure.hips_client import (
    get_image_url_from_hips_id,
    survey_to_hips_id,
)
from packages.galaxy_core.infrastructure.irsa_finderchart_client import fetch_2mass_cutout_jpeg
from packages.galaxy_core.infrastructure.mast_hst_client import (
    format_hst_jwst_info,
    search_hst_jwst,
)
from packages.galaxy_core.infrastructure.simbad_client import format_object_info
from packages.galaxy_core.infrastructure.simbad_client import query_object as simbad_query_object

logger = logging.getLogger(__name__)

_IMAGE_DOWNLOAD_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Module-level helpers (stateless, no registry dependency)
# ---------------------------------------------------------------------------


def _build_fetch_attempts(opts: dict[str, Any]) -> list[tuple[str | None, str | None]]:
    catalog_opt = opts.get("catalog")
    if catalog_opt:
        return [(str(catalog_opt).strip(), None)]
    band_opt = opts.get("band")
    if not band_opt:
        return [("SDSS", None)]
    band_str = str(band_opt).strip()
    if band_str.lower() in ("visible", "optical"):
        return [("SDSS", None)]
    return [(None, band_str)]


def _resolve_target(request: AnalyzeRequest, opts: dict[str, Any]) -> ResolvedTarget:
    """Resolve the request to a ResolvedTarget using the first successful attempt."""
    size_opt: float
    try:
        size_opt = float(opts.get("size_arcmin", 10.0))
    except (TypeError, ValueError):
        size_opt = 10.0

    ra_opt = opts.get("ra_deg")
    dec_opt = opts.get("dec_deg")

    if ra_opt is not None and dec_opt is not None:
        try:
            ra_val = float(ra_opt)
            dec_val = float(dec_opt)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid coordinates: ra_deg={ra_opt!r}, dec_deg={dec_opt!r}"
            ) from exc
        attempts = _build_fetch_attempts(opts)
        catalog_p, band_p = attempts[0]
        return resolve_and_fetch(
            ra_deg=ra_val, dec_deg=dec_val, catalog=catalog_p, band=band_p, size_arcmin=size_opt
        )

    name = (request.target and request.target.name or "").strip()
    if not name:
        raise ValueError("Target name is empty. Provide target.name or options ra_deg/dec_deg.")

    attempts = _build_fetch_attempts(opts)
    errors: list[str] = []
    for catalog_p, band_p in attempts:
        try:
            return resolve_and_fetch(
                name=name, catalog=catalog_p, band=band_p, size_arcmin=size_opt
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(str(exc))
    raise RuntimeError(f"Failed to resolve '{name}': " + "; ".join(errors))


def _download_image(
    request_id: str, resolved: ResolvedTarget, artifact_store: ArtifactWriter
) -> str:
    """Download the image for a ResolvedTarget and save it. Returns the artifact path."""
    if resolved.survey_used.upper() == "2MASS-J":
        jpg_bytes = fetch_2mass_cutout_jpeg(
            ra_deg=resolved.ra_deg,
            dec_deg=resolved.dec_deg,
            subsetsize_arcmin=resolved.size_arcmin,
        )
        return artifact_store.save_image(request_id, jpg_bytes).path

    if (dss2_plate := dss2_survey_to_plate(resolved.survey_used)) is not None:
        jpg_bytes = fetch_dss2_cutout_jpeg(
            ra_deg=resolved.ra_deg,
            dec_deg=resolved.dec_deg,
            size_arcmin=resolved.size_arcmin,
            plate=dss2_plate,
        )
        return artifact_store.save_image(request_id, jpg_bytes).path

    resp = _http.get(resolved.image_url, timeout=_IMAGE_DOWNLOAD_TIMEOUT)
    resp.raise_for_status()
    return artifact_store.save_image(request_id, resp.content).path


def _annotate_image_in_place(
    image_path: str,
    image: Any,
    mask: Any,
    measurements: dict[str, Any],
) -> None:
    """Overwrite the saved image with segmentation contour and centroid marker."""
    import io
    from pathlib import Path

    from PIL import Image as PILImage
    from PIL import ImageDraw

    img_min = float(image.min())
    img_max = float(image.max())
    scale = 255.0 / (img_max - img_min + 1e-9)
    img_u8 = ((image - img_min) * scale).astype(np.uint8)
    img_rgb = PILImage.fromarray(img_u8).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    mask_ratio = float(mask.sum()) / mask.size
    if mask_ratio < 0.5:
        from scipy.ndimage import binary_closing, gaussian_filter
        from skimage.measure import find_contours

        closed = binary_closing(mask.astype(bool), iterations=3)
        smooth = gaussian_filter(closed.astype(float), sigma=3.0)
        for contour in find_contours(smooth, level=0.5):  # type: ignore[no-untyped-call]
            if len(contour) < 15:
                continue
            pts = [(int(round(x)), int(round(y))) for y, x in contour]
            draw.line(pts, fill=(0, 220, 100), width=2)

    h, w = mask.shape
    cx = int(round(float(measurements.get("centroid_x", w / 2))))
    cy = int(round(float(measurements.get("centroid_y", h / 2))))
    r = max(4, min(12, w // 30))
    draw.line([(cx - r, cy), (cx + r, cy)], fill=(255, 130, 0), width=2)
    draw.line([(cx, cy - r), (cx, cy + r)], fill=(255, 130, 0), width=2)

    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=92)
    Path(image_path).write_bytes(buf.getvalue())


# ---------------------------------------------------------------------------
# Registry key helpers
# ---------------------------------------------------------------------------


def _artifacts_key(rid: str) -> str:
    return f"artifacts:{rid}"


def _results_key(rid: str) -> str:
    return f"results:{rid}"


def _image_path_key(rid: str) -> str:
    return f"image_path:{rid}"


def _coordinates_key(rid: str) -> str:
    return f"coordinates:{rid}"


def _morphology_key(rid: str) -> str:
    return f"morphology_text:{rid}"


def _object_info_key(rid: str) -> str:
    return f"object_info:{rid}"


def _object_info_text_key(rid: str) -> str:
    return f"object_info_text:{rid}"


def _hst_jwst_key(rid: str) -> str:
    return f"hst_jwst:{rid}"


def _hst_jwst_text_key(rid: str) -> str:
    return f"hst_jwst_text:{rid}"


def _summary_key(rid: str) -> str:
    return f"summary:{rid}"


def _get_artifacts(registry: ContextRegistry) -> list[Artifact]:
    try:
        return registry.get(_artifacts_key(registry.request_id))  # type: ignore[return-value]
    except KeyError:
        arts: list[Artifact] = []
        registry.put(_artifacts_key(registry.request_id), arts)
        return arts


def _get_results(registry: ContextRegistry) -> dict[str, Any]:
    try:
        return registry.get(_results_key(registry.request_id))  # type: ignore[return-value]
    except KeyError:
        res: dict[str, Any] = {}
        registry.put(_results_key(registry.request_id), res)
        return res


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------


def make_tool_resolve_and_fetch_image(
    registry: ContextRegistry,
    artifact_store: ArtifactWriter,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def resolve_and_fetch_image() -> dict[str, Any]:
        """Resolve the astronomical target and acquire the image.

        Handles three acquisition paths automatically:
        1. Viewer canvas image_data (base64 JPEG) — decode and save directly.
        2. Active viewer with HiPS ID — download frame via hips2fits API.
        3. Full resolve pipeline — SESAME name/coordinate resolution + survey download
           with automatic fallback chain (SDSS → DSS2 → GALEX → 2MASS → HiPS).

        This tool MUST always be called first. It stores the image array in the
        context registry and returns an image_handle string for subsequent tools.
        """
        opts = dict(request.options) if request.options else {}
        artifacts = _get_artifacts(registry)
        has_viewer = request.view_ra_deg is not None and request.view_dec_deg is not None
        rid = registry.request_id

        # For resolve task: only SESAME lookup, no image download regardless of viewer state.
        # The frontend uses the returned coordinates to open the Aladin viewer.
        # Must check before the viewer paths so a viewer-active request to "show M51"
        # resolves M51's coordinates instead of reusing the visible M87 canvas.
        if request.task == "resolve":
            resolved = _resolve_target(request, opts)
            coordinates: dict[str, Any] = {
                "ra_deg": resolved.ra_deg,
                "dec_deg": resolved.dec_deg,
                "survey_used": resolved.survey_used,
                "hips_id": survey_to_hips_id(resolved.survey_used),
                "size_arcmin": resolved.size_arcmin,
            }
            registry.put(_coordinates_key(rid), coordinates)
            return {
                "image_handle": "none",
                "survey_used": resolved.survey_used,
                "ra_deg": resolved.ra_deg,
                "dec_deg": resolved.dec_deg,
            }

        if has_viewer and request.image_data:
            raw = request.image_data
            if "," in raw:
                raw = raw.split(",", 1)[1]
            image_bytes = base64.b64decode(raw)
            artifact = artifact_store.save_image(rid, image_bytes)
            artifacts.append(artifact)
            image_path = artifact.path
            size = request.view_size_arcmin or 10.0
            coordinates = {
                "ra_deg": request.view_ra_deg,
                "dec_deg": request.view_dec_deg,
                "survey_used": request.view_hips_id,
                "hips_id": request.view_hips_id,
                "size_arcmin": size,
            }

        elif has_viewer and request.view_hips_id:
            size = request.view_size_arcmin or 10.0
            remote_url = get_image_url_from_hips_id(
                request.view_ra_deg,  # type: ignore[arg-type]
                request.view_dec_deg,  # type: ignore[arg-type]
                request.view_hips_id,
                size,
            )
            resp = _http.get(remote_url, timeout=_IMAGE_DOWNLOAD_TIMEOUT)
            resp.raise_for_status()
            artifact = artifact_store.save_image(rid, resp.content)
            artifacts.append(artifact)
            image_path = artifact.path
            coordinates = {
                "ra_deg": request.view_ra_deg,
                "dec_deg": request.view_dec_deg,
                "survey_used": request.view_hips_id,
                "hips_id": request.view_hips_id,
                "size_arcmin": size,
            }

        else:
            resolved = _resolve_target(request, opts)
            image_path = _download_image(rid, resolved, artifact_store)
            artifacts.append(Artifact(type="image", path=image_path))
            coordinates = {
                "ra_deg": resolved.ra_deg,
                "dec_deg": resolved.dec_deg,
                "survey_used": resolved.survey_used,
                "hips_id": survey_to_hips_id(resolved.survey_used),
                "size_arcmin": resolved.size_arcmin,
            }

        image_array = load_image(image_path)
        handle = registry.image_handle()
        registry.put(handle, image_array)
        registry.put(_image_path_key(rid), image_path)
        registry.put(_coordinates_key(rid), coordinates)

        return {
            "image_handle": handle,
            "image_path": image_path,
            "survey_used": str(coordinates.get("survey_used") or "unknown"),
            "ra_deg": coordinates.get("ra_deg"),
            "dec_deg": coordinates.get("dec_deg"),
        }

    return resolve_and_fetch_image


def make_tool_segment_image(
    registry: ContextRegistry,
    analyzer: BasicGalaxyAnalyzer,
    artifact_store: ArtifactWriter,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def segment_image(image_handle: str) -> dict[str, Any]:
        """Segment the galaxy in the loaded image to detect the primary source.

        Uses SEP (Source Extractor Python) background subtraction and source detection.
        Must be called after resolve_and_fetch_image using the returned image_handle.
        Stores the SegmentationResult in the context registry for subsequent analysis tools.
        Returns seg_handle and mask statistics.
        """
        image: np.ndarray = registry.get(image_handle)  # type: ignore[assignment]
        opts = request.options or {}
        thresh_sigma = float(opts.get("thresh_sigma", 2.0))

        segmentation = tool_segment(analyzer, image, thresh_sigma=thresh_sigma)

        artifacts = _get_artifacts(registry)
        artifacts.append(artifact_store.save_mask(request.request_id, segmentation.mask))

        results = _get_results(registry)
        results["segmentation_metadata"] = segmentation.metadata

        handle = registry.seg_handle()
        registry.put(handle, segmentation)

        mask_ratio = float(segmentation.mask.sum()) / max(segmentation.mask.size, 1)
        return {
            "seg_handle": handle,
            "mask_ratio": round(mask_ratio, 4),
            "detected": mask_ratio > 0.001,
        }

    return segment_image


def make_tool_analyze(
    registry: ContextRegistry,
    analyzer: BasicGalaxyAnalyzer,
    artifact_store: ArtifactWriter,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def analyze_galaxy(image_handle: str, seg_handle: str) -> dict[str, Any]:
        """Run morphological analysis on the segmented galaxy.

        Computes measurements based on the task type from the current request:
        - measure_basic: area, centroid, ellipticity, mean intensity
        - morphology_summary: full CAS + Sérsic + radial profile summary
        - cas: Concentration, Asymmetry, Smoothness parameters
        - radial_profile: brightness profile by annuli
        - sersic: Sérsic index fitting

        Must be called after segment_image. Uses image_handle and seg_handle from
        their respective tools. Stores metrics in the registry.
        Returns metrics dict and morphology_text summary.
        """
        image: np.ndarray = registry.get(image_handle)  # type: ignore[assignment]
        segmentation: SegmentationResult = registry.get(seg_handle)  # type: ignore[assignment]
        task = request.task or "measure_basic"
        opts = request.options or {}
        analysis_params = {k: opts[k] for k in ("n_bins",) if k in opts}

        analysis_results = tool_run_analysis(
            analyzer, image, segmentation, str(task), params=analysis_params or None
        )

        merged_metrics: dict[str, Any] = {}
        summary_parts: list[str] = []
        artifacts = _get_artifacts(registry)

        for ar in analysis_results:
            merged_metrics.update(ar.metrics)
            if ar.summary:
                summary_parts.append(ar.summary)
            if ar.image_png:
                artifacts.append(
                    artifact_store.save_plot(request.request_id, ar.module_name, ar.image_png)
                )

        if task == "measure_basic":
            mask = segmentation.mask
            merged_metrics.setdefault("area_pixels", float(mask.sum()))
            indices = np.argwhere(mask > 0)
            if indices.size > 0:
                merged_metrics.setdefault("centroid_x", float(indices[:, 1].mean()))
                merged_metrics.setdefault("centroid_y", float(indices[:, 0].mean()))
            else:
                merged_metrics.setdefault("centroid_x", 0.0)
                merged_metrics.setdefault("centroid_y", 0.0)
            merged_metrics.setdefault("ellipticity", 0.0)
            merged_metrics.setdefault(
                "mean_intensity",
                float(np.mean(image[mask > 0])) if mask.sum() > 0 else 0.0,
            )

        results = _get_results(registry)
        results["measurements"] = merged_metrics
        artifact_store.save_measurements(request.request_id, merged_metrics)

        # Annotate the saved image with contour and centroid
        try:
            image_path: str = registry.get(_image_path_key(registry.request_id))  # type: ignore[assignment]
            _annotate_image_in_place(image_path, image, segmentation.mask, merged_metrics)
        except (KeyError, Exception):
            logger.debug("annotate_image_skipped", exc_info=True)

        morphology_text = "\n\n".join(summary_parts)
        registry.put(_morphology_key(registry.request_id), morphology_text)
        registry.put(registry.metrics_handle(), merged_metrics)

        return {"metrics": merged_metrics, "morphology_text": morphology_text}

    return analyze_galaxy


def make_tool_run_isophotes(
    registry: ContextRegistry,
    analyzer: BasicGalaxyAnalyzer,
    artifact_store: ArtifactWriter,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def run_isophotes(image_handle: str, seg_handle: str) -> dict[str, Any]:
        """Fit elliptical isophotes to the detected galaxy using photutils.

        Traces concentric ellipses through the brightness distribution,
        producing an isophote table and an annotated PNG visualization.
        Best called after analyze_galaxy (for tasks morphology_summary or measure_basic)
        or after segment_image (for task isophotes).
        Returns isophote_summary with axis ratios, position angles, and intensity profiles.
        """
        image: np.ndarray = registry.get(image_handle)  # type: ignore[assignment]
        segmentation: SegmentationResult = registry.get(seg_handle)  # type: ignore[assignment]
        opts = request.options or {}
        target_name = (request.target and request.target.name) or "unknown"
        hips_id = opts.get("_resolved_hips_id") or request.view_hips_id
        n_iso = int(opts.get("n_iso", 8))

        # Use metrics already computed by analyze_galaxy if available,
        # avoiding a redundant full-morphology run just to get centroid/geometry.
        try:
            measurements: dict[str, Any] = registry.get(registry.metrics_handle())  # type: ignore[assignment]
        except KeyError:
            measurements = analyzer.measure_basic(image, segmentation)

        results = _get_results(registry)
        results["measurements"] = measurements

        try:
            _iso_table, png_bytes, iso_summary = tool_isophotes(
                image, segmentation, measurements, target_name, hips_id=hips_id, n_iso=n_iso
            )
        except Exception as exc:
            logger.warning("run_isophotes_failed: %s", exc, exc_info=True)
            iso_text = "Isofotas no disponibles para esta imagen."
            rid = registry.request_id
            try:
                existing_text = str(registry.get(_morphology_key(rid)))
                combined = f"{existing_text}\n\n{iso_text}" if existing_text else iso_text
            except KeyError:
                combined = iso_text
            registry.put(_morphology_key(rid), combined)
            return {"isophote_summary": iso_text}

        results["isophotes"] = _iso_table

        artifacts = _get_artifacts(registry)
        artifacts.append(artifact_store.save_plot(request.request_id, "isophotes", png_bytes))

        band = str(opts.get("band") or "visible")
        catalog_used = str(opts.get("catalog") or BAND_TO_SURVEY.get(band.lower(), "SDSS"))
        size_arcmin = float(opts.get("_resolved_size_arcmin") or opts.get("size_arcmin") or 10.0)
        iso_text = append_catalog_and_field(iso_summary, catalog_used, size_arcmin)

        # Append to existing analysis text instead of overwriting it.
        # For morphology_summary/measure_basic, analyze_galaxy already saved text here.
        rid = registry.request_id
        try:
            existing_text = str(registry.get(_morphology_key(rid)))
            combined = f"{existing_text}\n\n{iso_text}" if existing_text else iso_text
        except KeyError:
            combined = iso_text
        registry.put(_morphology_key(rid), combined)
        return {"isophote_summary": iso_text}

    return run_isophotes


def make_tool_enrich_metadata(
    registry: ContextRegistry,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def enrich_metadata() -> dict[str, Any]:
        """Enrich the analysis with external catalogue metadata.

        Queries two non-blocking external sources in sequence:
        - SIMBAD TAP: object type, morphological type, radial velocity, redshift.
        - MAST (HST/JWST): search for Hubble/Webb observations at the resolved position.

        Skip this tool for viewer-only requests without a named target.
        Returns object_info (SIMBAD) and hst_jwst (MAST) as formatted strings.
        Network failures are silently logged and return empty strings.
        """
        target_name = (request.target and request.target.name) or ""
        _PLACEHOLDER = "from conversation"
        object_info: dict[str, Any] | None = None
        hst_jwst: dict[str, Any] | None = None
        object_info_text = ""
        hst_jwst_text = ""

        if target_name and target_name != _PLACEHOLDER:
            try:
                simbad_record = simbad_query_object(target_name)
                if simbad_record:
                    object_info = simbad_record
                    object_info_text = format_object_info(simbad_record)
            except Exception:
                logger.warning("simbad_enrich_failed", exc_info=True)

        try:
            coords: dict[str, Any] = registry.get(_coordinates_key(registry.request_id))  # type: ignore[assignment]
            ra = coords.get("ra_deg")
            dec = coords.get("dec_deg")
            if ra is not None and dec is not None:
                hst_record = search_hst_jwst(float(ra), float(dec))
                if hst_record:
                    hst_jwst = hst_record
                    hst_jwst_text = format_hst_jwst_info(hst_record)
        except Exception:
            logger.warning("mast_hst_enrich_failed", exc_info=True)

        rid = registry.request_id
        if object_info is not None:
            registry.put(_object_info_key(rid), object_info)
        if object_info_text:
            registry.put(_object_info_text_key(rid), object_info_text)
        if hst_jwst is not None:
            registry.put(_hst_jwst_key(rid), hst_jwst)
        if hst_jwst_text:
            registry.put(_hst_jwst_text_key(rid), hst_jwst_text)

        return {"object_info": object_info_text, "hst_jwst": hst_jwst_text}

    return enrich_metadata


def make_tool_generate_final_report(
    registry: ContextRegistry,
    langchain_backend: Any,
    request: AnalyzeRequest,
) -> BaseTool:
    @tool
    def generate_final_report(
        morphology_text: str = "",
        object_info: str = "",
        hst_jwst_info: str = "",
    ) -> dict[str, Any]:
        """Generate the final narrative summary for the user.

        Combines morphological analysis text, SIMBAD metadata, and HST/JWST
        observations into a coherent scientific summary in Spanish.
        If no langchain_backend is available, returns morphology_text as-is.

        Pass the morphology_text from analyze_galaxy (or isophote_summary from
        run_isophotes), and the object_info / hst_jwst_info from enrich_metadata
        if those tools were called. This tool MUST always be called last.
        """
        opts = request.options or {}

        # Resolve task: SESAME-only — return viewer guidance so the user can explore first.
        if request.task == "resolve":
            target_name = (request.target and request.target.name) or None
            try:
                coords: dict[str, Any] = registry.get(_coordinates_key(registry.request_id))  # type: ignore[assignment]
                ra = coords.get("ra_deg")
                dec = coords.get("dec_deg")
                label = target_name or (
                    f"RA {ra:.4f}° Dec {dec:.4f}°" if ra and dec else "el objeto"
                )
            except KeyError:
                label = target_name or "el objeto"
            summary = (
                f"Aquí tienes {label}. "
                f"Ajusta el encuadre, el zoom y la banda en el visor como prefieras. "
                f"Cuando estés listo, dime qué quieres analizar: "
                f"morfología completa, segmentación e isofotas, "
                f"fotometría básica, o perfil de brillo."
            )
            registry.put(_summary_key(registry.request_id), summary)
            return {"summary": summary}

        # If the LLM didn't pass morphology_text, recover it from the registry
        # (saved there by analyze_galaxy or run_isophotes). This makes the tool
        # robust against the LLM omitting or truncating the argument.
        if not morphology_text:
            try:
                morphology_text = registry.get(_morphology_key(registry.request_id))  # type: ignore[assignment]
            except KeyError:
                pass

        # Tasks that produce no morphology text (segment, fetch_image): confirm completion
        # without returning the resolve viewer-guidance message.
        if not morphology_text:
            task = request.task or "segment"
            target_label = (request.target and request.target.name) or "el objeto"
            if task == "segment":
                summary = (
                    f"Segmentación de {target_label} completada. "
                    f"La máscara de la fuente detectada está guardada en los artefactos."
                )
            else:
                summary = f"Imagen de {target_label} cargada correctamente."
            registry.put(_summary_key(registry.request_id), summary)
            return {"summary": summary}

        if langchain_backend is None:
            summary = morphology_text
        else:
            simbad_record: dict[str, Any] = {}
            try:
                raw = registry.get(_object_info_key(registry.request_id))
                simbad_record = raw if isinstance(raw, dict) else {}
            except KeyError:
                pass

            summary = langchain_backend.generate_accompanying_summary(
                target_name=(request.target and request.target.name) or "unknown",
                band=str(opts.get("band") or "visible"),
                morphology_summary=morphology_text,
                user_message=last_user_message(request) or None,
                simbad_morph_type=simbad_record.get("morph_type") or None,
            )

        band = str(opts.get("band") or "visible")
        catalog_used = str(opts.get("catalog") or BAND_TO_SURVEY.get(band.lower(), "SDSS"))
        size_arcmin = float(opts.get("_resolved_size_arcmin") or opts.get("size_arcmin") or 10.0)
        summary = append_catalog_and_field(summary, catalog_used, size_arcmin)

        def _reg_text(key: str) -> str:
            try:
                val = registry.get(key)
                return str(val) if val else ""
            except KeyError:
                return ""

        rid = registry.request_id
        final_object_info = object_info or _reg_text(_object_info_text_key(rid))
        final_hst_jwst = hst_jwst_info or _reg_text(_hst_jwst_text_key(rid))

        if final_object_info:
            summary += f"\n\n{final_object_info}"
        if final_hst_jwst:
            summary += f"\n{final_hst_jwst}"

        registry.put(_summary_key(registry.request_id), summary)
        return {"summary": summary}

    return generate_final_report


# ---------------------------------------------------------------------------
# Public bundle builder (called by TaskOrchestrator)
# ---------------------------------------------------------------------------


def build_agent_tools(
    registry: ContextRegistry,
    analyzer: BasicGalaxyAnalyzer,
    artifact_store: ArtifactWriter,
    request: AnalyzeRequest,
    langchain_backend: Any,
) -> list[BaseTool]:
    """Return all six @tool instances bound to the given registry for one request."""
    return [
        make_tool_resolve_and_fetch_image(registry, artifact_store, request),
        make_tool_segment_image(registry, analyzer, artifact_store, request),
        make_tool_analyze(registry, analyzer, artifact_store, request),
        make_tool_run_isophotes(registry, analyzer, artifact_store, request),
        make_tool_enrich_metadata(registry, request),
        make_tool_generate_final_report(registry, langchain_backend, request),
    ]
