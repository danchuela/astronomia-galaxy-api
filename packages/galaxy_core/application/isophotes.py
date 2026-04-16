"""Isophote analysis: ellipse fitting (measurements) + isointensity contours (visualization)."""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NON_OPTICAL_KEYWORDS = ("galex", "nvss", "rass", "xmm", "radio", "fuv", "nuv")


def _survey_warning(hips_id: str | None) -> str | None:
    if not hips_id:
        return None
    low = hips_id.lower()
    if any(k in low for k in _NON_OPTICAL_KEYWORDS):
        return (
            "Nota: las isofotas elípticas asumen un perfil de brillo suave y simétrico. "
            "Los surveys UV (GALEX) y de radio muestran emisión irregular (regiones HII, "
            "anillos de formación estelar) que dificulta la convergencia. "
            "Para mejores resultados usa un survey óptico: SDSS, DSS2 o PanSTARRS."
        )
    return None


def _contour_levels(image: np.ndarray, mask: np.ndarray, n_levels: int = 8) -> list[float]:
    pixels = image[mask > 0] if mask.sum() > 0 else image.ravel()
    if pixels.size == 0:
        return []
    low = float(np.percentile(pixels, 15))
    high = float(np.percentile(pixels, 98))
    if high <= low or low <= 0:
        low = max(float(np.percentile(pixels, 5)), 1e-6)
    if high <= low:
        return []
    levels = np.logspace(np.log10(max(low, 1e-6)), np.log10(high), n_levels + 2)
    return levels[1:-1].tolist()  # type: ignore[no-any-return]


def _draw_contours(
    draw: Any,
    image: np.ndarray,
    levels: list[float],
    cx: float,
    cy: float,
    max_radius: float,
    color: tuple[int, int, int] = (0, 220, 100),
) -> int:
    """Draw isointensity contours, skipping outliers (field stars/artifacts); returns count."""
    from skimage.measure import find_contours

    if image.ndim < 2 or image.shape[0] < 2 or image.shape[1] < 2:
        return 0

    drawn = 0
    for level in levels:
        try:
            contours = find_contours(image, level)  # type: ignore[no-untyped-call]
        except Exception:
            continue
        for contour in contours:
            if len(contour) < 20:
                continue
            cont_cy = float(contour[:, 0].mean())
            cont_cx = float(contour[:, 1].mean())
            dist = np.hypot(cont_cx - cx, cont_cy - cy)
            if dist > max_radius:
                continue
            # skimage returns (row, col); PIL needs (x, y)
            pts = [(float(c[1]), float(c[0])) for c in contour]
            pts.append(pts[0])
            draw.line(pts, fill=color, width=1)
            drawn += 1
    return drawn


def _fit_ellipses(
    image: np.ndarray,
    cx: float,
    cy: float,
    ellip: float,
    pa: float,
    r_start: float,
    maxsma: float,
) -> list[dict[str, float]]:
    """Fit elliptical isophotes with photutils. Returns a list of per-isophote parameter dicts."""
    from photutils.isophote import Ellipse, EllipseGeometry

    img_fit = np.ascontiguousarray(np.clip(image, 0.0, None), dtype=np.float64)

    attempts = [
        {"sclip": 3.0, "n_clip": 2, "step": 0.2, "fix_center": True, "eps": ellip, "pa": pa},
        {"sclip": 5.0, "n_clip": 0, "step": 0.3, "fix_center": False, "eps": ellip, "pa": pa},
        {"sclip": 5.0, "n_clip": 0, "step": 0.4, "fix_center": False, "eps": 0.0, "pa": 0.0},
    ]

    for i, params in enumerate(attempts):
        try:
            sma0 = r_start if i < 2 else max(4.0, r_start * 0.5)
            geom = EllipseGeometry(x0=cx, y0=cy, sma=sma0, eps=params["eps"], pa=params["pa"])
            ell = Ellipse(img_fit, geometry=geom)
            result = ell.fit_image(
                sclip=params["sclip"],
                n_clip=params["n_clip"],
                step=params["step"],
                maxsma=maxsma,
                minsma=sma0,
                linear=False,
                fix_center=params["fix_center"],
            )
            if result is not None and len(result) >= 3:
                table = []
                for iso in result:
                    table.append(
                        {
                            "sma": round(float(iso.sma), 2),
                            "intens": round(float(iso.intens), 4),
                            "ellip": round(float(iso.eps), 4),
                            "pa_deg": round(float(np.degrees(iso.pa)), 2),
                            "x0": round(float(iso.x0), 2),
                            "y0": round(float(iso.y0), 2),
                        }
                    )
                return table
        except Exception:
            if i == len(attempts) - 1:
                logger.warning("photutils isophote fitting failed on all attempts", exc_info=True)

    return []


def compute_isophotes(
    image: np.ndarray,
    mask: np.ndarray,
    measurements: dict[str, Any],
    n_iso: int = 8,
    hips_id: str | None = None,
) -> tuple[list[dict[str, float]], bytes]:
    """Fit elliptical isophotes and render pixel-traced contours; returns (table, png_bytes)."""
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from scipy.ndimage import gaussian_filter

    h, w = image.shape
    if h < 4 or w < 4:
        img_u8 = ((np.clip(image, 0, None) / max(float(image.max()), 1e-9)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(img_u8).convert("RGB").save(buf, format="PNG")
        return [], buf.getvalue()

    cx = float(measurements.get("centroid_x", w / 2))
    cy = float(measurements.get("centroid_y", h / 2))
    indices = np.argwhere(mask > 0)
    if not (0 < cx < w and 0 < cy < h) and indices.size > 0:
        cy = float(indices[:, 0].mean())
        cx = float(indices[:, 1].mean())

    ellip = float(measurements.get("ellipticity", 0.2))
    ellip = min(max(ellip, 0.0), 0.85)

    rpetro = float(measurements.get("rpetro_circ", min(h, w) * 0.3))
    if not np.isfinite(rpetro) or rpetro <= 0:
        rpetro = min(h, w) * 0.3
    rpetro = min(rpetro, 0.45 * min(h, w))
    r_start = max(4.0, rpetro * 0.15)
    maxsma = min(rpetro * 1.1, 0.42 * min(h, w))
    maxsma = max(maxsma, r_start * 2)

    pa = float(measurements.get("orientation_asymmetry", 0.0))

    iso_table = _fit_ellipses(image, cx, cy, ellip, pa, r_start, maxsma)

    img_clipped = np.clip(image, 0.0, None)
    img_min, img_max = float(img_clipped.min()), float(img_clipped.max())
    scale = 255.0 / (img_max - img_min + 1e-9)
    img_u8 = ((img_clipped - img_min) * scale).astype(np.uint8)
    img_rgb = PILImage.fromarray(img_u8).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # Light smoothing removes pixel-level noise before contouring
    smooth = gaussian_filter(img_clipped.astype(np.float64), sigma=2.0)
    levels = _contour_levels(smooth, mask, n_iso)
    max_radius = max(maxsma * 1.5, min(h, w) * 0.45)
    drawn = _draw_contours(draw, smooth, levels, cx, cy, max_radius)

    if drawn == 0:
        try:
            from scipy.ndimage import binary_closing
            from scipy.ndimage import label as scipy_label
            from skimage.measure import find_contours as _find_contours

            closed = binary_closing(mask.astype(bool), iterations=3)
            smooth_mask = gaussian_filter(closed.astype(float), sigma=3.0)
            if smooth_mask.shape[0] < 2 or smooth_mask.shape[1] < 2:
                raise ValueError("mask too small for contour fallback")
            labeled, n_comp = scipy_label(smooth_mask > 0.4)
            if n_comp > 1:
                cix = max(0, min(int(round(cx)), w - 1))
                ciy = max(0, min(int(round(cy)), h - 1))
                main_lbl = labeled[ciy, cix]
                if main_lbl > 0:
                    smooth_mask = smooth_mask * (labeled == main_lbl)
                else:
                    sizes = [int(np.sum(labeled == i)) for i in range(1, n_comp + 1)]
                    smooth_mask = smooth_mask * (labeled == (int(np.argmax(sizes)) + 1))
            for contour in _find_contours(smooth_mask, level=0.5):  # type: ignore[no-untyped-call]
                if len(contour) < 15:
                    continue
                pts = [(float(c[1]), float(c[0])) for c in contour]
                pts.append(pts[0])
                draw.line(pts, fill=(0, 220, 100), width=2)
        except Exception:
            logger.debug("isophotes_contour_fallback_failed", exc_info=True)

    r = max(4, min(12, w // 30))
    draw.line([(int(cx) - r, int(cy)), (int(cx) + r, int(cy))], fill=(255, 130, 0), width=2)
    draw.line([(int(cx), int(cy) - r), (int(cx), int(cy) + r)], fill=(255, 130, 0), width=2)

    buf = io.BytesIO()
    img_rgb.save(buf, format="PNG")
    return iso_table, buf.getvalue()


def format_isophotes_summary(
    iso_table: list[dict[str, float]],
    target_name: str,
    hips_id: str | None = None,
) -> str:
    warn = _survey_warning(hips_id)
    if not iso_table:
        base = (
            f"Contornos de isointensidad trazados para {target_name}. "
            "El ajuste elíptico no convergió; la imagen muestra los contornos reales "
            "de brillo de la galaxia."
        )
        return f"{base}\n\n{warn}" if warn else base

    innermost = iso_table[0]
    outermost = iso_table[-1]
    base = (
        f"Isofotas para {target_name}: {len(iso_table)} elipses ajustadas "
        f"(sma={innermost['sma']:.1f}–{outermost['sma']:.1f} px, "
        f"elipticidad media {sum(r['ellip'] for r in iso_table) / len(iso_table):.3f}). "
        "La imagen muestra contornos de isointensidad reales que trazan la forma "
        "de la galaxia incluyendo brazos espirales y estructura irregular."
    )
    return f"{base}\n\n{warn}" if warn else base
