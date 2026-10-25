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
        converged = (
            all(sm.get(key) is not None for key in ("sersic_n", "sersic_rhalf", "sersic_ellip"))
            and sm.get("flag_sersic") == 0
        )
        keys = (
            "sersic_n",
            "sersic_rhalf",
            "sersic_ellip",
            "sersic_theta",
            "sersic_amplitude",
            "sersic_xc",
            "sersic_yc",
            "sersic_chi2_dof",
            "sersic_psf_corrected",
            "flag_sersic",
        )
        metrics = {key: sm.get(key) for key in keys}
        return AnalysisResult(
            module_name=self.name,
            metrics=metrics,
            summary=_sersic_summary(metrics, converged),
            image_png=_plot_sersic_residual(sm.get("_sersic_diagnostic")) if converged else None,
            status=(
                "ok"
                if converged and not sm.get("flag")
                else "suspect" if converged else "unavailable"
            ),
            warnings=[] if converged else ["invalid_sersic_fit"],
        )


def _sersic_summary(metrics: dict[str, Any], converged: bool) -> str:
    if not converged:
        return "Sérsic fit unavailable: no valid converged model."
    return (
        f"Sérsic fit: Sérsic index n={metrics['sersic_n']:.2f}, "
        f"r_half={metrics['sersic_rhalf']:.1f} px, ellipticity={metrics['sersic_ellip']:.3f}."
    )


def _plot_sersic_residual(diagnostic: dict[str, Any] | None) -> bytes | None:
    """Render the exact statmorph fitted stamp, not a new approximate model."""
    if diagnostic is None:
        return None
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm

    data, model, sigma = (np.asarray(diagnostic[key]) for key in ("data", "model", "sigma"))
    mask = np.asarray(diagnostic["mask"], dtype=bool)
    valid = ~mask & np.isfinite(data) & np.isfinite(model) & np.isfinite(sigma) & (sigma > 0)
    if not valid.any():
        return None
    residual = data - model
    standardized = np.divide(residual, sigma, out=np.zeros_like(residual), where=valid)
    low, high = np.percentile(data[valid], [1, 99])
    scale = Normalize(vmin=low, vmax=max(high, low + 1e-9))
    residual_limit = max(float(np.percentile(np.abs(residual[valid]), 99)), 1e-9)
    sigma_limit = max(float(np.percentile(np.abs(standardized[valid]), 99)), 1.0)
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    fig.patch.set_facecolor("#0f172a")
    for ax, values, title, cmap, norm in zip(
        axes,
        [data, model, residual, standardized],
        ["Data (background-sub)", "Fitted model", "Data - model", "Residual / sigma"],
        ["gray", "gray", "RdBu_r", "RdBu_r"],
        [
            scale,
            scale,
            TwoSlopeNorm(0, -residual_limit, residual_limit),
            TwoSlopeNorm(0, -sigma_limit, sigma_limit),
        ],
        strict=True,
    ):
        shown = ax.imshow(np.ma.array(values, mask=~valid), origin="upper", cmap=cmap, norm=norm)
        ax.set_title(title, fontsize=9, color="white")
        ax.axis("off")
        fig.colorbar(shown, ax=ax, shrink=0.7).ax.tick_params(labelsize=7, colors="white")
    fig.tight_layout(pad=0.7)
    output = io.BytesIO()
    fig.savefig(output, format="png", dpi=110, facecolor=fig.get_facecolor())
    plt.close(fig)
    return output.getvalue()
