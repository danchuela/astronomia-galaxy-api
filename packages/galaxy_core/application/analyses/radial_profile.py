from __future__ import annotations

import io
from typing import Any

import numpy as np

from packages.galaxy_core.application.morphology import compute_radial_profile
from packages.galaxy_core.domain.analysis import AnalysisContext, AnalysisResult


class RadialProfileAnalysis:
    name = "radial_profile"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        mask = ctx.segmentation.mask
        indices = np.argwhere(mask > 0)

        if indices.size == 0:
            return AnalysisResult(
                module_name=self.name,
                metrics={"radial_profile": {"radii_px": [], "mean_flux": []}},
                summary="Radial profile unavailable (empty mask).",
            )

        cx = float(indices[:, 1].mean())
        cy = float(indices[:, 0].mean())

        n_bins = int(ctx.params.get("n_bins", 25))
        profile = compute_radial_profile(ctx.image, mask, cx, cy, n_bins=n_bins)
        image_png = _plot_radial_profile(profile["radii_px"], profile["mean_flux"])

        return AnalysisResult(
            module_name=self.name,
            metrics={"radial_profile": profile},
            summary=_radial_summary(profile),
            image_png=image_png,
        )


def _radial_summary(profile: dict[str, Any]) -> str:
    radii = profile.get("radii_px", [])
    fluxes = profile.get("mean_flux", [])
    if not radii:
        return "Radial brightness profile: no data."
    r_max = radii[-1]
    peak_flux = max(fluxes) if fluxes else 0.0
    return (
        f"Radial brightness profile: {len(radii)} annuli out to r={r_max:.1f} px. "
        f"Peak mean flux: {peak_flux:.2f}."
    )


def _plot_radial_profile(radii: list[float], fluxes: list[float]) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if radii and fluxes:
        ax.plot(radii, fluxes, "o-", color="#38bdf8", lw=2, ms=4, markerfacecolor="#0ea5e9")
        ax.fill_between(radii, fluxes, alpha=0.15, color="#38bdf8")

        if len(fluxes) >= 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(fluxes, kernel, mode="valid")
            r_smooth = radii[2 : 2 + len(smoothed)]
            ax.plot(r_smooth, smoothed, "-", color="#f59e0b", lw=1.5, alpha=0.8, label="5-bin avg")
            ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    ax.set_xlabel("Radius (px)", fontsize=11, color="white")
    ax.set_ylabel("Mean Flux", fontsize=11, color="white")
    ax.set_title("Radial Brightness Profile", fontsize=12, color="white", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.grid(True, alpha=0.2, color="#475569")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
