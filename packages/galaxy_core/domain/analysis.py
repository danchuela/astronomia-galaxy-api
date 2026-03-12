from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np

from packages.galaxy_core.domain.models import SegmentationResult


@dataclass
class AnalysisResult:
    module_name: str
    metrics: dict[str, Any]
    summary: str
    image_png: bytes | None = None


@dataclass
class AnalysisContext:
    """Shared context for a single image analysis run; caches statmorph to avoid recomputation."""

    image: np.ndarray
    segmentation: SegmentationResult
    params: dict[str, Any] = field(default_factory=dict)
    _cache: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def cache_get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def cache_set(self, key: str, value: Any) -> None:
        self._cache[key] = value


@runtime_checkable
class GalaxyAnalysisModule(Protocol):
    """Protocol for analysis modules. Add new modules via AnalysisRegistry.register()."""

    name: ClassVar[str]

    def run(self, ctx: AnalysisContext) -> AnalysisResult: ...
