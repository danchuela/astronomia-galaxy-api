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


class GalaxyAnalyzer(Protocol):
    def segment_galaxy(self, image: np.ndarray) -> SegmentationResult: ...

    def measure_basic(
        self, image: np.ndarray, segmentation: SegmentationResult
    ) -> dict[str, Any]: ...

    def morphology_summary(self, measurements: dict[str, Any]) -> str: ...
