from __future__ import annotations

import io
from typing import Any

import numpy as np

from packages.galaxy_core.application.analyses.base import get_statmorph
from packages.galaxy_core.domain.analysis import AnalysisContext, AnalysisResult


class SersicAnalysis:
    name = "sersic"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        sm = get_statmorph(ctx)
        flag_sersic = sm.get("flag_sersic")
        converged = flag_sersic is not None and int(flag_sersic) == 0

        metrics: dict[str, Any] = {
            "sersic_n": sm.get("sersic_n") if converged else None,
            "sersic_rhalf": sm.get("sersic_rhalf") if converged else None,
            "sersic_ellip": sm.get("sersic_ellip") if converged else None,
            "sersic_theta": sm.get("sersic_theta") if converged else None,
            "flag_sersic": flag_sersic,
        }

        image_png: bytes | None = None
        if converged and ctx.segmentation.data_sub is not None:
            image_png = _plot_sersic_residual(
                ctx.segmentation.data_sub,
                ctx.segmentation.mask,
                metrics,
            )

        return AnalysisResult(
            module_name=self.name,
            metrics=metrics,
            summary=_sersic_summary(metrics, converged),
            image_png=image_png,
        )


def _sersic_summary(m: dict[str, Any], converged: bool) -> str:
    if not converged:
        return "Sérsic fit did not converge (flag_sersic > 0)."
    n = m.get("sersic_n")
    rhalf = m.get("sersic_rhalf")
    ellip = m.get("sersic_ellip")
    parts = []
    if n is not None:
        parts.append(f"Sérsic index n={n:.2f}")
    if rhalf is not None:
        parts.append(f"r_half={rhalf:.1f} px")
    if ellip is not None:
        parts.append(f"ellipticity={ellip:.3f}")
    return ("Sérsic fit: " + ", ".join(parts) + ".") if parts else "Sérsic fit: no parameters."


def _plot_sersic_residual(
    data_sub: np.ndarray,
    mask: np.ndarray,
    metrics: dict[str, Any],
) -> bytes | None:
    """Three-panel figure: background-subtracted data, Sérsic model, residual."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from astropy.modeling.models import Sersic2D
        from matplotlib.colors import Normalize

        rhalf = metrics.get("sersic_rhalf") or 1.0
        n = metrics.get("sersic_n") or 1.0
        ellip = metrics.get("sersic_ellip") or 0.0
        theta = metrics.get("sersic_theta") or 0.0

        h, w = data_sub.shape
        yy, xx = np.mgrid[0:h, 0:w]
        cx = w / 2.0
        cy = h / 2.0

        amplitude = float(np.median(data_sub[mask > 0])) if mask.sum() > 0 else 1.0
        model = Sersic2D(
            amplitude=amplitude,
            r_eff=max(rhalf, 1.0),
            n=max(n, 0.1),
            x_0=cx,
            y_0=cy,
            ellip=min(ellip, 0.99),
            theta=theta,
        )
        model_image = model(xx, yy).astype(np.float32)
        residual = data_sub - model_image

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
        fig.patch.set_facecolor("#0f172a")

        titles = ["Data (background-sub)", "Sérsic Model", "Residual (data − model)"]
        images = [data_sub, model_image, residual]
        cmaps = ["gray", "gray", "RdBu_r"]

        for ax, img, title, cmap in zip(axes, images, titles, cmaps, strict=False):
            ax.set_facecolor("#1e293b")
            norm = Normalize(vmin=np.percentile(img, 1), vmax=np.percentile(img, 99))
            ax.imshow(img, origin="upper", cmap=cmap, norm=norm)
            ax.set_title(title, fontsize=9, color="white", pad=4)
            ax.axis("off")

        fig.suptitle(
            f"Sérsic 2D Fit  (n={n:.2f}, r_half={rhalf:.1f} px)", fontsize=11, color="white", y=1.02
        )
        plt.tight_layout(pad=0.5)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        return None
