"""Validate the capture contract without silently downloading a different field."""

import math
from typing import Any

from packages.galaxy_agent.models import AnalyzeRequest
from packages.galaxy_core.domain.errors import ImageIntegrityError


def validate_view_capture(request: AnalyzeRequest, shape: tuple[int, ...] | None = None) -> None:
    meta: dict[str, Any] = request.view_metadata
    if not meta:
        return  # Older clients do not supply capture metadata.
    if not request.image_data or meta.get("capture_required"):
        raise ImageIntegrityError("A fresh viewer capture is required")
    if (
        request.view_ra_deg is None
        or request.view_dec_deg is None
        or request.view_size_arcmin is None
    ):
        raise ImageIntegrityError("Viewer field geometry is required")
    try:
        ra, dec, fov = (float(meta[key]) for key in ("ra_deg", "dec_deg", "fov_x_deg"))
        fov_y = float(meta["fov_y_deg"])
        width, height = int(meta["width_px"]), int(meta["height_px"])
        if not all(math.isfinite(value) for value in (ra, dec, fov, fov_y)):
            raise ValueError("Nonfinite capture geometry")
        delta_ra = abs((ra - float(request.view_ra_deg) + 180) % 360 - 180)
        if (
            delta_ra > 1e-7
            or abs(dec - float(request.view_dec_deg)) > 1e-7
            or abs(fov * 60 - float(request.view_size_arcmin)) > 1e-5
            or meta["hips_id"] != request.view_hips_id
            or meta["projection"] != "SIN"
            or not (0 < fov <= 180 and 0 < fov_y <= 180)
            or width <= 0
            or height <= 0
            or (shape is not None and shape != (height, width))
        ):
            raise ValueError("Capture geometry mismatch")
    except (TypeError, ValueError, KeyError) as exc:
        raise ImageIntegrityError("Viewer pixels and field metadata do not match") from exc
