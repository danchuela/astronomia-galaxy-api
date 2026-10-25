from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass
class SegmentationResult:
    mask: np.ndarray
    metadata: dict[str, float | int | str]
    data_sub: np.ndarray | None = None  # background-subtracted image (SEP)
    segmap_labels: np.ndarray | None = None  # grown label map — for visual contours
    tight_segmap_labels: np.ndarray | None = None  # raw SEP labels — for statmorph
    main_label: int = 1
    background: np.ndarray | None = None
    noise_sigma: np.ndarray | None = None
    psf: np.ndarray | None = None

    @property
    def excluded_mask(self) -> np.ndarray:
        labels = self.tight_segmap_labels
        if labels is None:
            labels = self.segmap_labels
        excluded = np.zeros(self.mask.shape, dtype=bool)
        if labels is not None:
            excluded |= (labels > 0) & (labels != self.main_label)
        if self.data_sub is not None:
            excluded |= ~np.isfinite(self.data_sub)
        if self.noise_sigma is not None:
            excluded |= ~np.isfinite(self.noise_sigma) | (self.noise_sigma <= 0)
        return excluded


class GalaxyAnalyzer(Protocol):
    def segment_galaxy(self, image: np.ndarray) -> SegmentationResult: ...

    def measure_basic(
        self, image: np.ndarray, segmentation: SegmentationResult
    ) -> dict[str, Any]: ...

    def morphology_summary(self, measurements: dict[str, Any]) -> str: ...
