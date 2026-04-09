from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def compute_sep_background(image: np.ndarray) -> tuple[np.ndarray, float]:
    """Background-subtract with SEP. Returns (data_sub, global_rms)."""
    import sep

    data = np.ascontiguousarray(image, dtype=np.float32)
    # SEP requires native byte order; force copy if big-endian
    if not data.dtype.isnative:
        data = data.byteswap().newbyteorder()  # type: ignore[attr-defined]
    h, w = data.shape
    # Large boxes for extended sources — prevents subtracting galaxy flux
    bw = max(64, w // 4)
    bh = max(64, h // 4)
    bkg = sep.Background(data, bw=bw, bh=bh)
    data_sub = np.ascontiguousarray(data - bkg, dtype=np.float32)
    return data_sub, float(bkg.globalrms)


def compute_sep_segmap(
    data_sub: np.ndarray, rms: float, thresh: float = 2.0
) -> tuple[Any, np.ndarray, np.ndarray, int]:
    """Detect sources with SEP; returns (objects, tight_segmap, grown_segmap, main_label)."""
    import sep
    from scipy.ndimage import binary_dilation

    err = max(rms, 1e-6)
    objects, segmap = sep.extract(
        data_sub,
        thresh=thresh,
        err=err,
        segmentation_map=True,
        deblend_nthresh=32,
        deblend_cont=0.05,
    )
    if len(objects) == 0:
        raise ValueError("SEP found no sources in the image")

    h, w = data_sub.shape
    cy, cx = h / 2.0, w / 2.0
    dist2 = (objects["x"] - cx) ** 2 + (objects["y"] - cy) ** 2
    score = objects["npix"].astype(float) / (dist2 + 1.0)
    main_idx = int(np.argmax(score))
    main_label = main_idx + 1  # SEP labels start at 1

    tight_segmap = segmap.astype(np.int32)

    # Grow pixels bordering the detected region to include faint outer flux (3px, 1.5σ threshold)
    main_mask = tight_segmap == main_label
    grow_thresh = 1.5 * err
    low_signal = data_sub > grow_thresh
    grown_pixels = binary_dilation(main_mask, iterations=3) & low_signal & (tight_segmap <= 0)
    grown_segmap = tight_segmap.copy()
    grown_segmap[grown_pixels] = main_label

    return objects, tight_segmap, grown_segmap, main_label


def compute_radial_profile(
    image: np.ndarray,
    mask: np.ndarray,
    cx: float,
    cy: float,
    n_bins: int = 20,
) -> dict[str, list[float]]:
    """Radial brightness profile: mean flux in concentric annuli; returns radii_px and mean_flux."""
    h, w = image.shape
    yy, xx = np.mgrid[0:h, 0:w]
    radii_map = np.sqrt((xx.astype(float) - cx) ** 2 + (yy.astype(float) - cy) ** 2)
    r_max = float(radii_map[mask > 0].max()) if mask.sum() > 0 else float(radii_map.max())
    edges = np.linspace(0.0, r_max, n_bins + 1)

    radii_out: list[float] = []
    flux_out: list[float] = []
    for i in range(n_bins):
        ring = (radii_map >= edges[i]) & (radii_map < edges[i + 1])
        pixels = image[ring]
        if pixels.size > 0:
            radii_out.append(float((edges[i] + edges[i + 1]) / 2.0))
            flux_out.append(float(pixels.mean()))

    return {"radii_px": radii_out, "mean_flux": flux_out}


def _make_gaussian_psf(fwhm: float = 2.0, size: int = 11) -> np.ndarray:
    """Normalized synthetic Gaussian PSF for statmorph."""
    sigma = fwhm / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    y, x = np.mgrid[-(size // 2) : (size // 2) + 1, -(size // 2) : (size // 2) + 1]
    psf = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float64)
    return psf / psf.sum()  # type: ignore[no-any-return]


def compute_statmorph_metrics(
    data_sub: np.ndarray,
    segmap_labels: np.ndarray,
    main_label: int,
    background_rms: float = 0.0,
) -> dict[str, Any]:
    """Full morphological metrics via statmorph: CAS, Sérsic, radial shape."""
    import statmorph

    # Single-source segmap avoids the 'label' conflict in source_morphology 0.7.x
    single_segmap = (segmap_labels == main_label).astype(np.int32)
    psf = _make_gaussian_psf(fwhm=2.0)
    # Inverse-variance weight map estimated from SEP background RMS
    rms = max(background_rms, 1e-6)
    weightmap = np.full(data_sub.shape, 1.0 / rms**2, dtype=np.float64)
    source_morph = statmorph.source_morphology(
        data_sub, single_segmap, psf=psf, weightmap=weightmap
    )
    morph = source_morph[0]

    def _safe(val: Any) -> float | None:
        try:
            v = float(val)
            return None if (math.isnan(v) or math.isinf(v)) else v
        except (TypeError, ValueError):
            return None

    sersic_ok = int(morph.flag_sersic) == 0
    return {
        "concentration": _safe(morph.concentration),
        "asymmetry": _safe(morph.asymmetry),
        "smoothness": _safe(morph.smoothness),
        "ellipticity_asymmetry": _safe(morph.ellipticity_asymmetry),
        "elongation_asymmetry": _safe(morph.elongation_asymmetry),
        "orientation_asymmetry": _safe(morph.orientation_asymmetry),
        "centroid_x": _safe(morph.xc_asymmetry),
        "centroid_y": _safe(morph.yc_asymmetry),
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
    }


def morphology_to_text(measurements: dict[str, Any]) -> str:
    area = measurements.get("area_pixels", 0.0)
    frame_too_small = measurements.get("frame_too_small", False)

    lines: list[str] = [f"Detected structure area: ~{area:.0f} px²."]

    if frame_too_small:
        lines.append(
            "The galaxy fills the entire field of view. Zoom in to isolate the galaxy"
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
