"""Isophote analysis: ellipse fitting (measurements) + isointensity contours (visualization)."""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_NON_OPTICAL_KEYWORDS = ("galex", "nvss", "rass", "xmm", "radio", "fuv", "nuv")


def _survey_warning(hips_id: str | None, lang: str = "es") -> str | None:
    if not hips_id:
        return None
    low = hips_id.lower()
    if any(k in low for k in _NON_OPTICAL_KEYWORDS):
        if lang == "en":
            return (
                "Note: elliptical isophotes assume a smooth, symmetric brightness profile. "
                "UV (GALEX) and radio surveys show irregular emission (HII regions, "
                "star-forming rings) that makes convergence difficult. "
                "For better results use an optical survey: SDSS, DSS2 or PanSTARRS."
            )
        return (
            "Nota: las isofotas elípticas asumen un perfil de brillo suave y simétrico. "
            "Los surveys UV (GALEX) y de radio muestran emisión irregular (regiones HII, "
            "anillos de formación estelar) que dificulta la convergencia. "
            "Para mejores resultados usa un survey óptico: SDSS, DSS2 o PanSTARRS."
        )
    return None


def _contour_levels(image: np.ndarray, mask: np.ndarray, n_levels: int = 8) -> list[float]:
    pixels = image[mask > 0] if mask.sum() > 0 else image.ravel()
    pixels = pixels[np.isfinite(pixels)]
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
    excluded_mask: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Fit elliptical isophotes with photutils. Returns a list of per-isophote parameter dicts."""
    from photutils.isophote import Ellipse, EllipseGeometry

    invalid = ~np.isfinite(image)
    if excluded_mask is not None:
        invalid |= excluded_mask
    img_fit = np.ma.array(np.asarray(image, dtype=np.float64), mask=invalid)

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
                    if (
                        not iso.valid
                        or not np.isfinite([iso.sma, iso.intens, iso.eps, iso.pa]).all()
                    ):
                        continue

                    def error(name: str, iso: Any = iso) -> float | None:
                        value = getattr(iso, name, None)
                        return float(value) if value is not None and np.isfinite(value) else None

                    table.append(
                        {
                            "sma": float(iso.sma),
                            "intens": float(iso.intens),
                            "ellip": float(iso.eps),
                            "pa_deg": float(np.degrees(iso.pa)),
                            "x0": float(iso.x0),
                            "y0": float(iso.y0),
                            "intens_err": error("int_err"),
                            "ellip_err": error("ellip_err"),
                            "pa_err_rad": error("pa_err"),
                            "stop_code": int(iso.stop_code),
                        }
                    )
                if len(table) >= 3:
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
    excluded_mask: np.ndarray | None = None,
) -> tuple[list[dict[str, Any]], bytes]:
    """Fit elliptical isophotes and render pixel-traced contours; returns (table, png_bytes)."""
    from PIL import Image as PILImage
    from PIL import ImageDraw
    from scipy.ndimage import gaussian_filter

    if not 3 <= n_iso <= 32:
        raise ValueError("n_iso must be between 3 and 32")
    h, w = image.shape
    if h < 4 or w < 4:
        img_u8 = ((np.clip(image, 0, None) / max(float(image.max()), 1e-9)) * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(img_u8).convert("RGB").save(buf, format="PNG")
        return [], buf.getvalue()

    center_x, center_y = measurements.get("centroid_x"), measurements.get("centroid_y")
    cx = float(center_x if center_x is not None else w / 2)
    cy = float(center_y if center_y is not None else h / 2)
    indices = np.argwhere(mask > 0)
    if not (0 < cx < w and 0 < cy < h) and indices.size > 0:
        cy = float(indices[:, 0].mean())
        cx = float(indices[:, 1].mean())

    ellip = float(
        measurements.get("ellipticity") or measurements.get("ellipticity_asymmetry") or 0.0
    )
    ellip = min(max(ellip, 0.0), 0.85)

    rpetro = float(measurements.get("rpetro_circ") or min(h, w) * 0.3)
    if not np.isfinite(rpetro) or rpetro <= 0:
        rpetro = min(h, w) * 0.3
    rpetro = min(rpetro, 0.45 * min(h, w))
    r_start = max(4.0, rpetro * 0.15)
    maxsma = min(rpetro * 1.1, 0.42 * min(h, w))
    maxsma = max(maxsma, r_start * 2)

    pa = float(
        measurements.get("orientation_centroid") or measurements.get("orientation_asymmetry") or 0.0
    )

    iso_table = _fit_ellipses(image, cx, cy, ellip, pa, r_start, maxsma, excluded_mask)

    img_clipped = np.maximum(np.where(np.isfinite(image), image, 0.0), 0.0)
    img_min, img_max = float(img_clipped.min()), float(img_clipped.max())
    scale = 255.0 / (img_max - img_min + 1e-9)
    img_u8 = ((img_clipped - img_min) * scale).astype(np.uint8)
    img_rgb = PILImage.fromarray(img_u8).convert("RGB")
    draw = ImageDraw.Draw(img_rgb)

    # Masked smoothing prevents detected neighbors from contributing contours.
    valid = np.isfinite(image)
    if excluded_mask is not None:
        valid &= ~excluded_mask
    numerator = gaussian_filter(np.where(valid, img_clipped, 0.0), sigma=2.0)
    denominator = gaussian_filter(valid.astype(float), sigma=2.0)
    smooth = np.divide(
        numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0
    )
    smooth[~valid] = np.nan
    levels = _contour_levels(smooth, mask, n_iso)
    max_radius = max(maxsma * 1.5, min(h, w) * 0.45)
    _draw_contours(draw, smooth, levels, cx, cy, max_radius)

    r = max(4, min(12, w // 30))
    draw.line([(int(cx) - r, int(cy)), (int(cx) + r, int(cy))], fill=(255, 130, 0), width=2)
    draw.line([(int(cx), int(cy) - r), (int(cx), int(cy) + r)], fill=(255, 130, 0), width=2)

    buf = io.BytesIO()
    img_rgb.save(buf, format="PNG")
    return iso_table, buf.getvalue()


def format_isophotes_summary(
    iso_table: list[dict[str, Any]],
    target_name: str,
    hips_id: str | None = None,
    lang: str = "es",
) -> str:
    warn = _survey_warning(hips_id, lang)
    if not iso_table and lang == "en":
        base = (
            f"Isointensity contours visualization for {target_name}. "
            "No valid elliptical isophote fit is available. "
            "Any displayed contours are smoothed intensity contours, not fitted ellipses."
        )
        return f"{base}\n\n{warn}" if warn else base
    if not iso_table:
        base = (
            f"Visualización de isointensidad para {target_name}. "
            "No hay un ajuste elíptico válido. Los contornos que puedan mostrarse son "
            "de intensidad suavizada, no elipses ajustadas."
        )
        return f"{base}\n\n{warn}" if warn else base

    innermost = iso_table[0]
    outermost = iso_table[-1]
    if lang == "en":
        base = (
            f"Isophotes for {target_name}: {len(iso_table)} fitted ellipses "
            f"(sma={innermost['sma']:.1f}–{outermost['sma']:.1f} px, "
            f"mean ellipticity {sum(r['ellip'] for r in iso_table) / len(iso_table):.3f}). "
            "The visualization shows smoothed intensity contours, "
            "distinct from the fitted ellipses."
        )
        return f"{base}\n\n{warn}" if warn else base
    base = (
        f"Isofotas para {target_name}: {len(iso_table)} elipses ajustadas "
        f"(sma={innermost['sma']:.1f}–{outermost['sma']:.1f} px, "
        f"elipticidad media {sum(r['ellip'] for r in iso_table) / len(iso_table):.3f}). "
        "La visualización muestra contornos de intensidad suavizada, "
        "distintos de las elipses ajustadas."
    )
    return f"{base}\n\n{warn}" if warn else base
