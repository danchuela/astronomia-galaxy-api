"""Code-owned measurement completeness and input limitations, never inferred by the LLM."""

from typing import Any

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
    issues = [
        "rendered_image",
        "background_noise_only",
        "unknown_psf",
        "no_parameter_uncertainties",
    ]
    segmentation = results.get("segmentation_metadata", {})
    if segmentation.get("touches_boundary"):
        issues.append("source_touches_boundary")
    if segmentation.get("n_sources", 0) > 1:
        issues.append("multiple_sources")
    if any(state == "suspect" for state in modules.values()):
        issues.append("suspect_measurement")
    return {
        "status": status,
        "input_kind": "rendered",
        "calibrated": False,
        "psf": "not_supplied",
        "uncertainty": "background_only",
        "issues": issues,
        "modules": modules,
    }


def quality_summary(quality: dict[str, Any], lang: str = "es") -> str:
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
    return prefix + (
        "Exploratory analysis of the rendered image, not calibrated photometry."
        if lang == "en"
        else "Análisis exploratorio de la imagen renderizada, no fotometría calibrada."
    )
