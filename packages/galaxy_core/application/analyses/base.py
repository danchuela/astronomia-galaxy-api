from __future__ import annotations

import logging
from typing import Any

from packages.galaxy_core.application.morphology import compute_statmorph_metrics
from packages.galaxy_core.domain.analysis import AnalysisContext

logger = logging.getLogger(__name__)

_STATMORPH_KEY = "statmorph"


def get_statmorph(ctx: AnalysisContext) -> dict[str, Any]:
    """Return statmorph metrics, computing at most once per context (lazy cache)."""
    cached = ctx.cache_get(_STATMORPH_KEY)
    if cached is not None:
        return cached  # type: ignore[no-any-return]

    seg = ctx.segmentation
    statmorph_segmap = (
        seg.tight_segmap_labels if seg.tight_segmap_labels is not None else seg.segmap_labels
    )
    if seg.data_sub is None or statmorph_segmap is None:
        result: dict[str, Any] = {}
    else:
        rms = float(seg.metadata.get("background_rms", 0.0))
        try:
            result = compute_statmorph_metrics(
                seg.data_sub,
                statmorph_segmap,
                seg.main_label,
                background_rms=rms,
            )
        except Exception as exc:
            logger.warning("statmorph_failed_in_module", extra={"error": str(exc)}, exc_info=True)
            result = {}

    ctx.cache_set(_STATMORPH_KEY, result)
    return result
