from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer
from packages.galaxy_core.domain import SegmentationResult
from packages.galaxy_core.domain.analysis import AnalysisResult

logger = logging.getLogger(__name__)


def load_image(image_url: str | None, timeout_seconds: int = 15) -> np.ndarray:
    if image_url is None:
        raise ValueError("An actual input image is required")

    if image_url.startswith(("http://", "https://")):
        response = requests.get(image_url, timeout=timeout_seconds)
        response.raise_for_status()
        data = response.content
    else:
        path = image_url.removeprefix("file://")
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        data = resolved.read_bytes()

    image: Image.Image = Image.open(io.BytesIO(data))
    if image.mode not in ("I", "F", "I;16", "I;16B", "I;16L"):
        image = image.convert("L")
    return np.asarray(image, dtype=np.float32)


def tool_segment(
    analyzer: BasicGalaxyAnalyzer, image: np.ndarray, thresh_sigma: float = 2.0
) -> SegmentationResult:
    return analyzer.segment_galaxy(image, thresh_sigma=thresh_sigma)


def tool_run_analysis(
    analyzer: BasicGalaxyAnalyzer,
    image: np.ndarray,
    segmentation: SegmentationResult,
    task: str,
    params: dict[str, object] | None = None,
) -> list[AnalysisResult]:
    return analyzer.run_task(task, image, segmentation, params=params)


def tool_isophotes(
    image: np.ndarray,
    segmentation: SegmentationResult,
    measurements: dict[str, Any],
    target_name: str = "unknown",
    hips_id: str | None = None,
    n_iso: int = 8,
    lang: str = "es",
) -> tuple[list[dict[str, Any]], bytes, str]:
    """Fit isophotes with photutils. Returns (table, png_bytes, summary)."""
    from packages.galaxy_core.application.isophotes import (
        compute_isophotes,
        format_isophotes_summary,
    )

    if segmentation.data_sub is None:
        raise ValueError("Isophotes require a background-subtracted image")
    data = segmentation.data_sub
    iso_table, png_bytes = compute_isophotes(
        data,
        segmentation.mask,
        measurements,
        n_iso=n_iso,
        hips_id=hips_id,
        excluded_mask=segmentation.excluded_mask,
    )
    summary = format_isophotes_summary(iso_table, target_name, hips_id=hips_id, lang=lang)
    return iso_table, png_bytes, summary
