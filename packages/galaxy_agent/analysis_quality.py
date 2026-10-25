"""Code-owned measurement completeness and input limitations, never inferred by the LLM."""

from typing import Any

from packages.galaxy_agent.messages import image_capacity_message
from packages.galaxy_core.application.analyses.registry import TASK_MODULES


def required_modules(task: str | None) -> list[str]:
    if task == "segment":
        return ["segment"]
    if task == "isophotes":
        return ["isophotes"]
    modules = list(TASK_MODULES.get(task or "", []))
    if task in ("morphology_summary", "measure_basic"):
        modules.append("isophotes")
    return modules


def build_analysis_quality(results: dict[str, Any], task: str | None) -> dict[str, Any]:
    recorded = results.get("modules", {})
    modules = {
        name: recorded.get(name, {}).get("status", "missing") for name in required_modules(task)
    }
    available = sum(status in ("ok", "suspect") for status in modules.values())
    status = "exploratory"
    if modules and available < len(modules):
        status = "partial" if available else "unavailable"
    provenance = results.get("image_provenance", {})
    native_fits = provenance.get("input_kind") == "fits"
    supplied_sigma = provenance.get("noise_model") == "user_sigma"
    supplied_psf = provenance.get("psf") == "user_supplied"
    stability = results.get("noise_stability", {})
    has_stability = stability.get("status") == "available"
    issues = [
        "unverified_calibration" if native_fits else "rendered_image",
    ]
    if any("resource_limit" in item.get("warnings", []) for item in recorded.values()):
        issues.append("resource_limit")
    if not supplied_sigma:
        issues.append("background_noise_only")
    if not supplied_psf:
        issues.append("unknown_psf")
    issues.append("conditional_noise_stability" if has_stability else "no_parameter_uncertainties")
    segmentation = results.get("segmentation_metadata", {})
    if segmentation.get("touches_boundary"):
        issues.append("source_touches_boundary")
    if segmentation.get("n_sources", 0) > 1:
        issues.append("multiple_sources")
    if segmentation.get("selection_method") == "user_point":
        issues.append("user_selected_source")
    if segmentation.get("region_applied"):
        issues.append("user_selected_region")
    if any(state == "suspect" for state in modules.values()):
        issues.append("suspect_measurement")
    return {
        "status": status,
        "input_kind": "fits" if native_fits else "rendered",
        "source": provenance.get("source", "viewer"),
        "calibrated": False,
        "psf": "user_supplied" if supplied_psf else "not_supplied",
        "uncertainty": "user_sigma" if supplied_sigma else "background_only",
        "noise_stability": stability,
        "issues": issues,
        "modules": modules,
    }


def quality_summary(quality: dict[str, Any], lang: str = "es") -> str:
    if "resource_limit" in quality.get("issues", []):
        return image_capacity_message(lang)
    if quality["status"] == "unavailable":
        return (
            "No valid measurements were obtained for the requested analysis. "
            "Check the framing and source detection; no missing values have been filled in."
            if lang == "en"
            else "No se han obtenido mediciones válidas para el análisis solicitado. "
            "Revisa el encuadre y la detección de la fuente; "
            "no se han rellenado los valores ausentes."
        )
    prefix = (
        ("Partial results. " if lang == "en" else "Resultados parciales. ")
        if quality["status"] == "partial"
        else ""
    )
    if quality.get("input_kind") == "fits":
        return prefix + (
            "Exploratory analysis on native FITS pixels. "
            "Photometric calibration and PSF have not been verified."
            if lang == "en"
            else "Análisis exploratorio sobre los píxeles originales del FITS. "
            "La calibración fotométrica y la PSF no se han verificado."
        )
    return prefix + (
        "Exploratory analysis of the rendered image, not calibrated photometry."
        if lang == "en"
        else "Análisis exploratorio de la imagen renderizada, no fotometría calibrada."
    )
