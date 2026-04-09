from __future__ import annotations

import numpy as np

from packages.galaxy_core.domain.analysis import (
    AnalysisContext,
    AnalysisResult,
    GalaxyAnalysisModule,
)
from packages.galaxy_core.domain.models import SegmentationResult


class AnalysisRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, GalaxyAnalysisModule] = {}

    def register(self, module: GalaxyAnalysisModule) -> None:
        self._modules[module.name] = module

    def run(
        self,
        names: list[str],
        image: np.ndarray,
        segmentation: SegmentationResult,
        params: dict[str, object] | None = None,
    ) -> list[AnalysisResult]:
        """Run the named modules sharing a single AnalysisContext."""
        ctx = AnalysisContext(image=image, segmentation=segmentation, params=params or {})
        results: list[AnalysisResult] = []
        for name in names:
            module = self._modules.get(name)
            if module is not None:
                results.append(module.run(ctx))
        return results

def default_registry() -> AnalysisRegistry:
    """Build the registry with all standard analysis modules registered."""
    from packages.galaxy_core.application.analyses.cas import CASAnalysis
    from packages.galaxy_core.application.analyses.radial_profile import RadialProfileAnalysis
    from packages.galaxy_core.application.analyses.sersic import SersicAnalysis

    registry = AnalysisRegistry()
    registry.register(CASAnalysis())  # type: ignore[arg-type]
    registry.register(SersicAnalysis())  # type: ignore[arg-type]
    registry.register(RadialProfileAnalysis())  # type: ignore[arg-type]
    return registry


TASK_MODULES: dict[str, list[str]] = {
    "segment": [],
    "measure_basic": ["cas", "radial_profile"],
    "morphology_summary": ["cas", "sersic", "radial_profile"],
    "cas": ["cas"],
    "radial_profile": ["radial_profile"],
    "sersic": ["sersic"],
}
