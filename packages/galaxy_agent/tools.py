from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import numpy as np
import requests
from PIL import Image

from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer, create_synthetic_image
from packages.galaxy_core.domain import SegmentationResult
from packages.galaxy_core.domain.analysis import AnalysisResult

logger = logging.getLogger(__name__)


def load_image(image_url: str | None, timeout_seconds: int = 15) -> np.ndarray:
    if image_url is None:
        logger.warning("load_image called without URL; using synthetic image")
        return create_synthetic_image()

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

    image = Image.open(io.BytesIO(data)).convert("L")
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
) -> tuple[list[dict[str, float]], bytes, str]:
    """Fit isophotes with photutils. Returns (table, png_bytes, summary)."""
    from packages.galaxy_core.application.isophotes import (
        compute_isophotes,
        format_isophotes_summary,
    )

    # For heavily extended galaxies (>30% of frame) SEP background subtraction
    # produces negative pixels that destabilize isophote fitting; use the raw image instead.
    mask_ratio = float(segmentation.mask.sum()) / max(segmentation.mask.size, 1)
    if mask_ratio > 0.30 or segmentation.data_sub is None:
        data = image
    else:
        data = segmentation.data_sub
    iso_table, png_bytes = compute_isophotes(
        data, segmentation.mask, measurements, n_iso=n_iso, hips_id=hips_id
    )
    summary = format_isophotes_summary(iso_table, target_name, hips_id=hips_id)
    return iso_table, png_bytes, summary
