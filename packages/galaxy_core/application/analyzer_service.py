from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from packages.galaxy_core.application.analyses.registry import (
    TASK_MODULES,
    AnalysisRegistry,
    default_registry,
)
from packages.galaxy_core.application.morphology import (
    compute_radial_profile,
    compute_sep_background,
    compute_sep_segmap,
)
from packages.galaxy_core.domain.analysis import AnalysisResult
from packages.galaxy_core.domain.models import SegmentationResult
from packages.galaxy_core.infrastructure.synthetic import normalize_image

logger = logging.getLogger(__name__)


@dataclass
class BasicGalaxyAnalyzer:
    threshold_quantile: float = 0.75
    registry: AnalysisRegistry = field(default_factory=default_registry)

    def segment_galaxy(self, image: np.ndarray, thresh_sigma: float = 2.0) -> SegmentationResult:
        try:
            data_sub, rms = compute_sep_background(image)
            objects, tight_segmap, grown_segmap, main_label = compute_sep_segmap(
                data_sub, rms, thresh=thresh_sigma
            )
            mask = (grown_segmap == main_label).astype(np.uint8)
            metadata: dict[str, float | int | str] = {
                "algorithm": "sep",
                "n_sources": int(len(objects)),
                "main_label": main_label,
                "mask_pixels": int(mask.sum()),
                "background_rms": float(rms),
            }
            return SegmentationResult(
                mask=mask,
                metadata=metadata,
                data_sub=data_sub,
                segmap_labels=grown_segmap,
                tight_segmap_labels=tight_segmap,
                main_label=main_label,
            )
        except Exception as exc:
            logger.warning("sep_segmentation_failed", extra={"error": str(exc)}, exc_info=True)
            return self._fallback_segment(image)

    def _fallback_segment(self, image: np.ndarray) -> SegmentationResult:
        normalized = normalize_image(image)
        threshold = float(np.quantile(normalized, self.threshold_quantile))
        mask = (normalized >= threshold).astype(np.uint8)
        return SegmentationResult(
            mask=mask,
            metadata={
                "algorithm": "quantile_fallback",
                "threshold": threshold,
                "mask_pixels": int(mask.sum()),
            },
        )

    def run_task(
        self,
        task: str,
        image: np.ndarray,
        segmentation: SegmentationResult,
        params: dict[str, object] | None = None,
    ) -> list[AnalysisResult]:
        if task not in TASK_MODULES:
            raise ValueError(f"Unknown task: {task!r}. Valid tasks: {sorted(TASK_MODULES)}")
        modules = TASK_MODULES[task]
        return self.registry.run(modules, image, segmentation, params=params)

    def measure_basic(self, image: np.ndarray, segmentation: SegmentationResult) -> dict[str, Any]:
        """Compatibility wrapper: run all modules and flatten metrics into a single dict."""
        results = self.registry.run(TASK_MODULES["morphology_summary"], image, segmentation)
        merged: dict[str, Any] = {}
        for r in results:
            merged.update(r.metrics)

        if "radial_profile" not in merged:
            indices = np.argwhere(segmentation.mask > 0)
            if indices.size > 0:
                cx = float(indices[:, 1].mean())
                cy = float(indices[:, 0].mean())
                merged["radial_profile"] = compute_radial_profile(image, segmentation.mask, cx, cy)
            else:
                merged["radial_profile"] = {"radii_px": [], "mean_flux": []}

        merged.setdefault("area_pixels", float(segmentation.mask.sum()))
        merged.setdefault(
            "centroid_x",
            (
                float(np.argwhere(segmentation.mask > 0)[:, 1].mean())
                if segmentation.mask.sum() > 0
                else 0.0
            ),
        )
        merged.setdefault(
            "centroid_y",
            (
                float(np.argwhere(segmentation.mask > 0)[:, 0].mean())
                if segmentation.mask.sum() > 0
                else 0.0
            ),
        )
        merged.setdefault("ellipticity", 0.0)
        merged.setdefault(
            "mean_intensity",
            float(np.mean(image[segmentation.mask > 0])) if segmentation.mask.sum() > 0 else 0.0,
        )
        merged.setdefault("analysis_reliable", False)
        return merged

    def morphology_summary(self, measurements: dict[str, Any]) -> str:
        from packages.galaxy_core.application.morphology import morphology_to_text

        return morphology_to_text(measurements)
