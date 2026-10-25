from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

from packages.galaxy_core.domain.errors import NoSourceDetectedError
from packages.galaxy_core.domain.models import SegmentationResult

logger = logging.getLogger(__name__)


def compute_sep_background(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Background-subtract with SEP. Returns (data_sub, global_rms)."""
    data_sub, _background, sigma = compute_sep_background_maps(image)
    return data_sub, float(np.median(sigma))


def compute_sep_background_maps(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Preserve the spatial background and background-only sigma maps."""
    import sep

    if image.ndim != 2 or min(image.shape) < 4 or not np.isfinite(image).any():
        raise ValueError("Expected a nonempty two-dimensional image with finite pixels")
    data = np.ascontiguousarray(image, dtype=np.float64)
    invalid = ~np.isfinite(data)
    safe_data = np.where(invalid, np.nanmedian(np.where(invalid, np.nan, data)), data)
    h, w = data.shape
    # Large boxes for extended sources — prevents subtracting galaxy flux
    bw = max(64, w // 4)
    bh = max(64, h // 4)
    bkg = sep.Background(np.ascontiguousarray(safe_data), mask=invalid, bw=bw, bh=bh)
    background = np.asarray(bkg.back(), dtype=np.float64)
    sigma = np.asarray(bkg.rms(), dtype=np.float64)
    data_sub = np.ascontiguousarray(data - background, dtype=np.float64)
    return data_sub, background, sigma


def compute_sep_segmap(
    data_sub: np.ndarray,
    rms: float | np.ndarray,
    thresh: float = 2.0,
    *,
    source_point: tuple[float, float] | None = None,
    excluded_mask: np.ndarray | None = None,
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Detect sources with SEP; returns (objects, tight_segmap, grown_segmap, main_label)."""
    import sep
    from scipy.ndimage import binary_dilation

    if not np.isfinite(thresh) or thresh <= 0:
        raise ValueError("Detection threshold must be positive")
    sigma = np.broadcast_to(np.asarray(rms, dtype=float), data_sub.shape)
    invalid = ~np.isfinite(data_sub) | ~np.isfinite(sigma) | (sigma <= 0)
    if excluded_mask is not None:
        invalid |= excluded_mask
    if invalid.all():
        raise NoSourceDetectedError("No usable background noise estimate or source")
    err = np.ascontiguousarray(np.where(invalid, 1.0, sigma), dtype=np.float32)
    objects, segmap = sep.extract(
        np.ascontiguousarray(np.where(invalid, 0.0, data_sub), dtype=np.float32),
        thresh=thresh,
        err=err,
        mask=invalid,
        segmentation_map=True,
        deblend_nthresh=32,
        deblend_cont=0.05,
    )
    if len(objects) == 0:
        raise NoSourceDetectedError("SEP found no sources in the image")

    h, w = data_sub.shape
    cy, cx = h / 2.0, w / 2.0
    dist2 = (objects["x"] - cx) ** 2 + (objects["y"] - cy) ** 2
    score = objects["npix"].astype(float) / (dist2 + 1.0)
    main_idx = int(np.argmax(score))
    if source_point is not None:
        x, y = source_point
        if not (np.isfinite(x) and np.isfinite(y) and 0 <= x < w and 0 <= y < h):
            raise NoSourceDetectedError("Selected point is outside the image")
        label = int(segmap[min(h - 1, int(round(y))), min(w - 1, int(round(x)))])
        if label <= 0:
            raise NoSourceDetectedError("No detected source at the selected point")
        main_idx = label - 1
    main_label = main_idx + 1  # SEP labels start at 1

    tight_segmap = segmap.astype(np.int32)

    # Grow pixels bordering the detected region to include faint outer flux (3px, 1.5σ threshold)
    main_mask = tight_segmap == main_label
    grow_thresh = 1.5 * err
    low_signal = data_sub > grow_thresh
    grown_pixels = (
        binary_dilation(main_mask, iterations=3) & low_signal & (tight_segmap <= 0) & ~invalid
    )
    grown_segmap = tight_segmap.copy()
    grown_segmap[grown_pixels] = main_label

    return objects, tight_segmap, grown_segmap, main_label


def compute_radial_profile(
    image: np.ndarray,
    mask: np.ndarray,
    cx: float,
    cy: float,
    n_bins: int = 20,
    *,
    excluded_mask: np.ndarray | None = None,
    noise_sigma: np.ndarray | None = None,
    ellipticity: float = 0.0,
    theta: float = 0.0,
) -> dict[str, list[float]]:
    """Mean background-subtracted intensity in annuli, excluding contaminating pixels."""
    if not 4 <= n_bins <= 100:
        raise ValueError("n_bins must be between 4 and 100")
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dx, dy = xx.astype(float) - cx, yy.astype(float) - cy
    major = dx * np.cos(theta) + dy * np.sin(theta)
    minor = -dx * np.sin(theta) + dy * np.cos(theta)
    q = max(1.0 - float(ellipticity), 0.05)
    radii_map = np.hypot(major, minor / q)
    r_max = float(radii_map[mask > 0].max()) if mask.sum() > 0 else float(radii_map.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)

    radii_out: list[float] = []
    flux_out: list[float] = []
    errors_out: list[float] = []
    counts: list[float] = []
    good = np.isfinite(image)
    if excluded_mask is not None:
        good &= ~excluded_mask
    if noise_sigma is not None:
        good &= np.isfinite(noise_sigma) & (noise_sigma > 0)
    for i in range(n_bins):
        ring = (radii_map >= edges[i]) & (radii_map <= edges[i + 1])
        if i:
            ring &= radii_map > edges[i]
        valid = ring & good
        pixels = image[valid]
        if pixels.size > 0:
            radii_out.append(float((edges[i] + edges[i + 1]) / 2.0))
            flux_out.append(float(pixels.mean()))
            counts.append(float(pixels.size))
            if noise_sigma is not None:
                errors_out.append(float(np.sqrt(np.square(noise_sigma[valid]).sum()) / pixels.size))

    return {
        "radii_px": radii_out,
        "mean_flux": flux_out,
        "mean_flux_err": errors_out,
        "n_pixels": counts,
    }


def compute_basic_measurements(
    image: np.ndarray, segmentation: SegmentationResult
) -> dict[str, Any]:
    data = segmentation.data_sub if segmentation.data_sub is not None else image
    valid = (segmentation.mask > 0) & ~segmentation.excluded_mask & np.isfinite(data)
    if not valid.any() or float(data[valid].sum()) <= 0:
        raise NoSourceDetectedError("The detected region has no positive measurable flux")
    yy, xx = np.indices(data.shape, dtype=float)
    weights = np.where(valid, np.maximum(data, 0), 0)
    total = weights.sum()
    cx, cy = float((weights * xx).sum() / total), float((weights * yy).sum() / total)
    dx, dy = xx - cx, yy - cy
    covariance = (
        np.array(
            [
                [(weights * dx * dx).sum(), (weights * dx * dy).sum()],
                [(weights * dx * dy).sum(), (weights * dy * dy).sum()],
            ]
        )
        / total
    )
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    ellipticity = (
        float(1 - np.sqrt(max(eigenvalues[0], 0) / eigenvalues[1])) if eigenvalues[1] > 0 else None
    )
    vector = eigenvectors[:, 1]
    return {
        "area_pixels": float(valid.sum()),
        "centroid_x": cx,
        "centroid_y": cy,
        "ellipticity": ellipticity,
        "orientation_centroid": float(np.arctan2(vector[1], vector[0]) % np.pi),
        "mean_intensity": float(data[valid].mean()),
        "flux_sum": float(data[valid].sum()),
        "intensity_unit": "input_intensity",
        "analysis_reliable": False,
    }


def compute_statmorph_metrics(
    data_sub: np.ndarray,
    segmap_labels: np.ndarray,
    main_label: int,
    background_rms: float = 0.0,
    *,
    noise_sigma: np.ndarray | None = None,
    excluded_mask: np.ndarray | None = None,
    psf: np.ndarray | None = None,
) -> dict[str, Any]:
    """Full morphological metrics via statmorph: CAS, Sérsic, radial shape."""
    import statmorph

    sigma = (
        np.asarray(noise_sigma, dtype=float)
        if noise_sigma is not None
        else np.full(data_sub.shape, background_rms, dtype=float)
    )
    if sigma.shape != data_sub.shape or not np.any(np.isfinite(sigma) & (sigma > 0)):
        raise ValueError("A positive sigma image in input intensity units is required")
    mask = ~np.isfinite(data_sub) | ~np.isfinite(sigma) | (sigma <= 0)
    if excluded_mask is not None:
        mask |= excluded_mask
    if psf is not None:
        psf = np.asarray(psf, dtype=float)
        if psf.ndim != 2 or not np.isfinite(psf).all() or psf.sum() <= 0 or (psf < 0).any():
            raise ValueError("Invalid PSF")
        psf = psf / psf.sum()
    morph = statmorph.SourceMorphology(
        np.where(np.isfinite(data_sub), data_sub, 0.0),
        segmap_labels,
        main_label,
        mask=mask,
        psf=psf,
        weightmap=np.where(mask, 0.0, sigma),
    )

    def _safe(val: Any) -> float | None:
        try:
            v = float(val)
            return None if not math.isfinite(v) or v == -99.0 else v
        except (TypeError, ValueError):
            return None

    morph_ok = int(morph.flag) < 2
    sersic_ok = int(morph.flag_sersic) == 0 and morph_ok
    result: dict[str, Any] = {
        "concentration": _safe(morph.concentration),
        "asymmetry": _safe(morph.asymmetry),
        "smoothness": _safe(morph.smoothness),
        "ellipticity_asymmetry": _safe(morph.ellipticity_asymmetry),
        "elongation_asymmetry": _safe(morph.elongation_asymmetry),
        "orientation_asymmetry": _safe(morph.orientation_asymmetry),
        "asymmetry_center_x": _safe(morph.xc_asymmetry),
        "asymmetry_center_y": _safe(morph.yc_asymmetry),
        "rpetro_circ": _safe(morph.rpetro_circ),
        "r20": _safe(morph.r20),
        "r80": _safe(morph.r80),
        "sn_per_pixel": _safe(morph.sn_per_pixel),
        "flag": int(morph.flag),
        "flag_sersic": int(morph.flag_sersic),
        "sersic_n": _safe(morph.sersic_n) if sersic_ok else None,
        "sersic_rhalf": _safe(morph.sersic_rhalf) if sersic_ok else None,
        "sersic_ellip": _safe(morph.sersic_ellip) if sersic_ok else None,
        "sersic_theta": _safe(morph.sersic_theta) if sersic_ok else None,
        "sersic_amplitude": _safe(morph.sersic_amplitude) if sersic_ok else None,
        "sersic_xc": _safe(morph.sersic_xc) if sersic_ok else None,
        "sersic_yc": _safe(morph.sersic_yc) if sersic_ok else None,
        "sersic_chi2_dof": _safe(morph.sersic_chi2_dof) if sersic_ok else None,
        "sersic_psf_corrected": psf is not None,
    }
    if not morph_ok:
        for key in ("concentration", "asymmetry", "smoothness"):
            result[key] = None
    if sersic_ok:
        # statmorph 0.7.2 fits on this stamp. Reuse its fitted callable, including its PSF.
        stamp = morph._slice_stamp
        data = data_sub[stamp]
        yy, xx = np.indices(data.shape, dtype=float)
        result["_sersic_diagnostic"] = {
            "data": data,
            "model": np.asarray(morph._sersic_model(xx, yy)),
            "sigma": sigma[stamp],
            "mask": (mask | ((segmap_labels > 0) & (segmap_labels != main_label)))[stamp],
        }
    return result


def morphology_to_text(measurements: dict[str, Any]) -> str:
    area = measurements.get("area_pixels", 0.0)
    frame_too_small = measurements.get("frame_too_small", False)

    lines: list[str] = [f"Detected structure area: ~{area:.0f} px²."]

    if frame_too_small:
        lines.append(
            "The galaxy fills the entire field of view. Zoom out to isolate the galaxy"
            " from the background for a reliable morphological analysis."
        )

    c = measurements.get("concentration")
    a = measurements.get("asymmetry")
    s = measurements.get("smoothness")
    if c is not None:
        a_str = f"{a:.3f}" if a is not None else "N/A"
        s_str = f"{s:.3f}" if s is not None else "N/A"
        lines.append(f"CAS — Concentration: {c:.3f}, Asymmetry: {a_str}, Smoothness: {s_str}.")

    sn = measurements.get("sersic_n")
    if sn is not None:
        rhalf = measurements.get("sersic_rhalf")
        rhalf_str = f"{rhalf:.1f}" if rhalf is not None else "N/A"
        lines.append(f"Sérsic index n={sn:.2f} (r_half={rhalf_str} px).")

    rpetro = measurements.get("rpetro_circ")
    r20 = measurements.get("r20")
    r80 = measurements.get("r80")
    if rpetro is not None:
        r20_str = f"{r20:.1f}" if r20 is not None else "N/A"
        r80_str = f"{r80:.1f}" if r80 is not None else "N/A"
        lines.append(f"Petrosian radius: {rpetro:.1f} px; r20={r20_str} px, r80={r80_str} px.")

    ellip = measurements.get("ellipticity_asymmetry")
    if ellip is not None:
        lines.append(f"Ellipticity: {ellip:.3f}.")

    return " ".join(lines)


def basic_measurements_to_text(measurements: dict[str, Any], lang: str = "es") -> str:
    """Expose basic measurements in the report, not only in its JSON payload."""
    area = measurements["area_pixels"]
    cx, cy = measurements["centroid_x"], measurements["centroid_y"]
    ellip = measurements["ellipticity"]
    ellipticity = f"{ellip:.3f}" if ellip is not None else "N/A"
    mean, total = measurements["mean_intensity"], measurements["flux_sum"]
    if lang == "en":
        return (
            f"Segmented area: {area:.0f} px²; flux-weighted centroid (x, y): "
            f"({cx:.2f}, {cy:.2f}) px; "
            f"moment ellipticity: {ellipticity}. Mean background-subtracted intensity: {mean:.3f}; "
            f"summed intensity: {total:.3f} (input image units, not calibrated photometry)."
        )
    return (
        f"Área segmentada: {area:.0f} px²; centroide ponderado por intensidad (x, y): "
        f"({cx:.2f}, {cy:.2f}) px; "
        f"elipticidad por momentos: {ellipticity}. Intensidad media con fondo restado: {mean:.3f}; "
        f"suma de intensidades: {total:.3f} (unidades de imagen, no fotometría calibrada)."
    )
