#!/usr/bin/env python3
"""Diagnostic: run statmorph on a real galaxy and print all flags and metrics."""

from __future__ import annotations

import sys

import numpy as np

from packages.galaxy_agent.tools import load_image
from packages.galaxy_core.application.morphology import (
    compute_sep_background,
    compute_sep_segmap,
)


def main() -> None:
    import math

    import statmorph

    target = sys.argv[1] if len(sys.argv) > 1 else "NGC 7727"
    size = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0

    print(f"=== Diagnosing statmorph for {target} (FOV={size} arcmin) ===\n")

    from packages.galaxy_core.application.resolve_and_fetch_service import resolve_and_fetch

    catalog = sys.argv[3] if len(sys.argv) > 3 else "DSS2"
    result = resolve_and_fetch(target, size_arcmin=size, catalog=catalog)
    print(f"Resolved: RA={result.ra_deg:.4f} Dec={result.dec_deg:.4f}")
    print(f"Image URL: {result.image_url}")

    image = load_image(result.image_url)
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Image range: [{image.min():.1f}, {image.max():.1f}]")

    data_sub, rms = compute_sep_background(image)
    print(f"\nSEP background RMS: {rms:.4f}")
    print(f"data_sub range: [{data_sub.min():.1f}, {data_sub.max():.1f}]")
    neg_frac = (data_sub < 0).sum() / data_sub.size
    print(f"Negative pixels: {neg_frac * 100:.1f}%")

    objects, tight_segmap, _grown_segmap, main_label = compute_sep_segmap(data_sub, rms)
    print(f"\nSEP sources: {len(objects)}, main_label={main_label}")
    mask = (tight_segmap == main_label).astype(np.uint8)
    mask_ratio = mask.sum() / mask.size
    print(f"Mask pixels: {mask.sum()} ({mask_ratio * 100:.1f}% of frame)")

    single_segmap = (tight_segmap == main_label).astype(np.int32)
    print(f"Segmap unique labels: {np.unique(single_segmap).tolist()}")

    from packages.galaxy_core.application.morphology import _make_gaussian_psf

    psf = _make_gaussian_psf(fwhm=2.0)
    weightmap = np.full(data_sub.shape, 1.0 / max(rms, 1e-6) ** 2, dtype=np.float64)

    print("\nRunning statmorph...")
    source_morph = statmorph.source_morphology(
        data_sub, single_segmap, psf=psf, weightmap=weightmap
    )
    morph = source_morph[0]

    def _show(name: str, val: object) -> None:
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            print(f"  {name}: {val} *** NaN/Inf ***")
        else:
            print(f"  {name}: {val}")

    print("\n=== FLAGS ===")
    _show("flag", int(morph.flag))
    _show("flag_sersic", int(morph.flag_sersic))

    print("\n=== CAS ===")
    _show("concentration", morph.concentration)
    _show("asymmetry", morph.asymmetry)
    _show("smoothness", morph.smoothness)

    print("\n=== Gini / M20 ===")
    _show("gini", morph.gini)
    _show("m20", morph.m20)
    _show("gini_m20_merger", morph.gini_m20_merger)
    _show("gini_m20_bulge", morph.gini_m20_bulge)

    print("\n=== Petrosian / Size ===")
    _show("rpetro_circ", morph.rpetro_circ)
    _show("rpetro_ellip", morph.rpetro_ellip)
    _show("r20", morph.r20)
    _show("r50", morph.r50)
    _show("r80", morph.r80)
    _show("rmax_circ", morph.rmax_circ)

    print("\n=== Sérsic ===")
    _show("sersic_n", morph.sersic_n)
    _show("sersic_rhalf", morph.sersic_rhalf)
    _show("sersic_ellip", morph.sersic_ellip)

    print("\n=== Other ===")
    _show("sn_per_pixel", morph.sn_per_pixel)
    _show("xc_asymmetry", morph.xc_asymmetry)
    _show("yc_asymmetry", morph.yc_asymmetry)

    print("\n=== Diagnosis ===")
    flag = int(morph.flag)
    if flag & 1:
        print("  [!] Source extends to image edge")
    if flag & 2:
        print("  [!] Asymmetry center outside Petrosian aperture")
    if flag & 4:
        print("  [!] Segmap extraction failed")
    if flag == 0:
        print("  [OK] No flags")

    if math.isnan(morph.rpetro_circ):
        print("  [ROOT CAUSE] Petrosian radius failed → concentration, r20, r80 unavailable")
    if math.isnan(morph.asymmetry):
        print("  [ROOT CAUSE] Asymmetry minimization did not converge")
    if mask_ratio > 0.3:
        print(f"  [WARNING] Galaxy fills {mask_ratio * 100:.0f}% of frame — increase FOV")


if __name__ == "__main__":
    main()
