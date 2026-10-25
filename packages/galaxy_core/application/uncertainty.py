"""Conditional noise-perturbation stability, not total scientific error bars."""

from __future__ import annotations

from typing import Any

import numpy as np

from packages.galaxy_core.application.morphology import compute_statmorph_metrics
from packages.galaxy_core.domain.models import SegmentationResult

KEYS = ("concentration", "asymmetry", "smoothness", "sersic_n", "sersic_rhalf", "sersic_ellip")


def estimate_noise_stability(seg: SegmentationResult, samples: int = 16) -> dict[str, Any]:
    if (
        samples != 16
        or seg.data_sub is None
        or seg.noise_sigma is None
        or max(seg.mask.shape) > 512
    ):
        raise ValueError("Noise stability requires 16 samples and images at most 512 by 512")
    labels = seg.tight_segmap_labels if seg.tight_segmap_labels is not None else seg.segmap_labels
    if labels is None:
        raise ValueError("Segmentation labels are required")
    rng = np.random.default_rng(42)
    values: dict[str, list[float]] = {key: [] for key in KEYS}
    sigma = np.where(seg.excluded_mask, 0, seg.noise_sigma)
    failures = 0
    for _ in range(samples):
        trial = seg.data_sub + rng.normal(size=seg.data_sub.shape) * sigma
        try:
            metrics = compute_statmorph_metrics(
                trial,
                labels,
                seg.main_label,
                noise_sigma=seg.noise_sigma,
                excluded_mask=seg.excluded_mask,
                psf=seg.psf,
            )
        except (ValueError, RuntimeError, ArithmeticError):
            failures += 1
            continue
        for key in KEYS:
            value = metrics.get(key)
            if value is not None and np.isfinite(value):
                values[key].append(float(value))
    intervals = {}
    for key, measured in values.items():
        if len(measured) >= 12:
            p16, median, p84 = np.percentile(measured, [16, 50, 84])
            intervals[key] = {
                "std": float(np.std(measured, ddof=1)),
                "p16": float(p16),
                "median": float(median),
                "p84": float(p84),
                "valid_samples": len(measured),
            }
    return {
        "method": "independent_gaussian_noise_perturbation",
        "samples": samples,
        "seed": 42,
        "fixed": ["segmentation", "background", "psf"],
        "failed_samples": failures,
        "excludes": [
            "correlated_noise",
            "background_systematics",
            "psf_uncertainty",
            "calibration",
            "segmentation_uncertainty",
        ],
        "status": "available" if intervals else "unavailable",
        "intervals": intervals,
    }
