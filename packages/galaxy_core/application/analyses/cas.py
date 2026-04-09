from __future__ import annotations

import io
from typing import Any

from packages.galaxy_core.application.analyses.base import get_statmorph
from packages.galaxy_core.domain.analysis import AnalysisContext, AnalysisResult


class CASAnalysis:
    """CAS morphological indices (Conselice 2003): concentration, asymmetry, smoothness."""

    name = "cas"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        sm = get_statmorph(ctx)
        metrics: dict[str, Any] = {
            "concentration": sm.get("concentration"),
            "asymmetry": sm.get("asymmetry"),
            "smoothness": sm.get("smoothness"),
            "sn_per_pixel": sm.get("sn_per_pixel"),
            "flag": sm.get("flag"),
        }
        image_png = _plot_cas(metrics)
        return AnalysisResult(
            module_name=self.name,
            metrics=metrics,
            summary=_cas_summary(metrics),
            image_png=image_png,
        )


def _cas_summary(m: dict[str, Any]) -> str:
    c = m.get("concentration")
    if c is None:
        return "CAS metrics unavailable (statmorph could not process the image)."
    a = m.get("asymmetry")
    s = m.get("smoothness")
    a_s = f"{a:.3f}" if a is not None else "N/A"
    s_s = f"{s:.3f}" if s is not None else "N/A"
    return f"CAS — Concentration: {c:.3f}, Asymmetry: {a_s}, Smoothness: {s_s}."


def _plot_cas(m: dict[str, Any]) -> bytes | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    labels = ["Concentration (C)", "Asymmetry (A)", "Smoothness (S)"]
    keys = ["concentration", "asymmetry", "smoothness"]
    ref_ranges = [(2.0, 5.5), (0.0, 0.6), (0.0, 0.5)]
    values = [m.get(k) for k in keys]

    if all(v is None for v in values):
        return None

    fig, ax = plt.subplots(figsize=(6, 3.2))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    colors = ["#38bdf8", "#f59e0b", "#a78bfa"]
    y_pos = np.arange(len(labels))

    for i, (val, (rmin, rmax), color) in enumerate(zip(values, ref_ranges, colors, strict=False)):
        ax.barh(i, rmax - rmin, left=rmin, height=0.4, color=color, alpha=0.15)
        if val is not None:
            ax.barh(i, val, height=0.55, color=color, alpha=0.85)
            ax.text(
                val + 0.03,
                i,
                f"{val:.3f}",
                va="center",
                ha="left",
                color="white",
                fontsize=9,
                fontweight="bold",
            )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, color="white", fontsize=9)
    ax.set_xlabel("Value", color="white", fontsize=10)
    ax.set_title("CAS Indices", color="white", fontsize=11, pad=8)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.set_xlim(0, 6.5)
    ax.grid(axis="x", alpha=0.2, color="#475569")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
