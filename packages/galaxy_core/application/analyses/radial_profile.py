from __future__ import annotations

import io
from typing import Any

import numpy as np

from packages.galaxy_core.application.morphology import (
    compute_basic_measurements,
    compute_radial_profile,
)
from packages.galaxy_core.domain.analysis import AnalysisContext, AnalysisResult


class RadialProfileAnalysis:
    name = "radial_profile"

    def run(self, ctx: AnalysisContext) -> AnalysisResult:
        mask = ctx.segmentation.mask
        indices = np.argwhere(mask > 0)

        if indices.size == 0 or ctx.segmentation.data_sub is None:
            return AnalysisResult(
                module_name=self.name,
                metrics={"radial_profile": {"radii_px": [], "mean_flux": []}},
                summary="Radial profile unavailable (empty mask).",
                status="unavailable",
                warnings=["missing_background_subtracted_image"],
            )

        geometry = compute_basic_measurements(ctx.image, ctx.segmentation)
        cx, cy = geometry["centroid_x"], geometry["centroid_y"]

        n_bins = int(ctx.params.get("n_bins", 25))
        profile = compute_radial_profile(
            ctx.segmentation.data_sub,
            mask,
            cx,
            cy,
            n_bins=n_bins,
            excluded_mask=ctx.segmentation.excluded_mask,
            noise_sigma=ctx.segmentation.noise_sigma,
            ellipticity=geometry["ellipticity"] or 0.0,
            theta=geometry["orientation_centroid"],
        )
        image_png = _plot_radial_profile(
            profile["radii_px"], profile["mean_flux"], profile["mean_flux_err"]
        )

        return AnalysisResult(
            module_name=self.name,
            metrics={
                "radial_profile": profile,
                "radial_profile_geometry": "elliptical_moments",
                "radial_profile_error_model": "independent_background_sigma_only",
            },
            summary=_radial_summary(profile),
            image_png=image_png,
            status="ok" if profile["radii_px"] else "unavailable",
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


def _plot_radial_profile(
    radii: list[float],
    fluxes: list[float],
    errors: list[float] | None = None,
) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    if radii and fluxes:
        ax.errorbar(radii, fluxes, yerr=errors or None, fmt="o-", color="#38bdf8", lw=2, ms=4)
        ax.fill_between(radii, fluxes, alpha=0.15, color="#38bdf8")

        if len(fluxes) >= 5:
            kernel = np.ones(5) / 5
            smoothed = np.convolve(fluxes, kernel, mode="valid")
            r_smooth = radii[2 : 2 + len(smoothed)]
            ax.plot(r_smooth, smoothed, "-", color="#f59e0b", lw=1.5, alpha=0.8, label="5-bin avg")
            ax.legend(fontsize=8, facecolor="#1e293b", edgecolor="#475569", labelcolor="white")

    ax.set_xlabel("Semi-major radius (px)", fontsize=11, color="white")
    ax.set_ylabel("Mean background-subtracted intensity", fontsize=10, color="white")
    ax.set_title("Radial Brightness Profile", fontsize=12, color="white", pad=10)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.grid(True, alpha=0.2, color="#475569")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()
