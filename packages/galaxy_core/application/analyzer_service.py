from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from packages.galaxy_core.application.analyses.registry import (
    TASK_MODULES,
    AnalysisRegistry,
    default_registry,
)
from packages.galaxy_core.application.morphology import (
    compute_basic_measurements,
    compute_sep_background_maps,
    compute_sep_segmap,
)
from packages.galaxy_core.domain.analysis import AnalysisResult
from packages.galaxy_core.domain.models import SegmentationResult


@dataclass
class BasicGalaxyAnalyzer:
    registry: AnalysisRegistry = field(default_factory=default_registry)

    def segment_galaxy(self, image: np.ndarray, thresh_sigma: float = 2.0) -> SegmentationResult:
        data_sub, background, sigma = compute_sep_background_maps(image)
        objects, tight_segmap, grown_segmap, main_label = compute_sep_segmap(
            data_sub, sigma, thresh=thresh_sigma
        )
        mask = (grown_segmap == main_label).astype(np.uint8)
        return SegmentationResult(
            mask=mask,
            metadata={
                "algorithm": "sep",
                "n_sources": int(len(objects)),
                "main_label": main_label,
                "background_rms": float(np.median(sigma)),
                "mask_pixels": int(mask.sum()),
                "noise_model": "background_sigma_only",
                "selection_method": "area_over_center_distance",
                "touches_boundary": int(
                    mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()
                ),
            },
            data_sub=data_sub,
            background=background,
            noise_sigma=sigma,
            segmap_labels=grown_segmap,
            tight_segmap_labels=tight_segmap,
            main_label=main_label,
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
        merged = compute_basic_measurements(image, segmentation)
        for r in results:
            merged.update(r.metrics)
        merged.setdefault("radial_profile", {"radii_px": [], "mean_flux": []})
        return merged

    def morphology_summary(self, measurements: dict[str, Any]) -> str:
        from packages.galaxy_core.application.morphology import morphology_to_text

        return morphology_to_text(measurements)
