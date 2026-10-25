"""Preview and source selection without an LLM call or saved user artifacts."""

from __future__ import annotations

import base64
import io
import math
from typing import Any

from PIL import Image

from packages.galaxy_agent.uploads import PreparedUpload, UploadedImage, fits_hdus, prepare_upload
from packages.galaxy_core.application.morphology import (
    compute_sep_background_maps,
    compute_sep_segmap,
)
from packages.galaxy_core.domain.errors import NoSourceDetectedError


def segmentation_inputs(upload: UploadedImage, prepared: PreparedUpload) -> dict[str, Any]:
    h, w = prepared.pixels.shape
    result: dict[str, Any] = {"noise_sigma": prepared.noise_sigma, "psf": prepared.psf}
    if upload.source_point:
        result["source_point"] = (upload.source_point.x * (w - 1), upload.source_point.y * (h - 1))
    if upload.region:
        r = upload.region
        result["region"] = (
            int(r.left * w),
            int(r.top * h),
            min(w, math.ceil(r.right * w)),
            min(h, math.ceil(r.bottom * h)),
        )
    return result


def inspect_image(upload: UploadedImage) -> dict[str, Any]:
    prepared = prepare_upload(upload)
    data_sub, _, sigma = compute_sep_background_maps(prepared.pixels)
    if prepared.noise_sigma is not None:
        sigma = prepared.noise_sigma
    sources = []
    try:
        objects, _, _, main_label = compute_sep_segmap(data_sub, sigma)
        for i, obj in enumerate(objects[:500]):
            sources.append(
                {
                    "label": i + 1,
                    "x": float(obj["x"]),
                    "y": float(obj["y"]),
                    "area": int(obj["npix"]),
                    "automatic": i + 1 == main_label,
                }
            )
    except NoSourceDetectedError:
        pass
    with Image.open(io.BytesIO(prepared.preview)) as image:
        image.thumbnail((1024, 1024))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
    h, w = prepared.pixels.shape
    return {
        "width": w,
        "height": h,
        "preview": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(),
        "sources": sources,
        "hdus": fits_hdus(prepared.original) if prepared.provenance["input_kind"] == "fits" else [],
        "sha256": prepared.provenance["sha256"],
    }
