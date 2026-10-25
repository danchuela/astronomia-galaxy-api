"""Reproducible recovery controls and a real GALFIT example, without an LLM.

Run from the repository: python scripts/validate_science.py --output <folder>
Optional --galfit-example points to the official galfit-ex.tar.gz archive.
The example's feedme values are initial guesses, NOT reference truth.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
import time
import warnings
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.modeling.models import Sersic2D
from scipy.ndimage import convolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer
from packages.galaxy_core.application.morphology import compute_statmorph_metrics


def measure(data, psf=None, sigma=None):
    seg = BasicGalaxyAnalyzer().segment_galaxy(data, noise_sigma=sigma, psf=psf)
    metrics = compute_statmorph_metrics(
        seg.data_sub,
        seg.tight_segmap_labels,
        seg.main_label,
        noise_sigma=seg.noise_sigma,
        excluded_mask=seg.excluded_mask,
        psf=psf,
    )
    return {key: value for key, value in metrics.items() if not key.startswith("_")}, seg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--galfit-example", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    controls = []
    for n in (1.0, 2.5, 4.0):
        for resolution, noise in ((1, 1.0), (2, 1.0), (1, 12.0)):
            size = 192 * resolution
            yy, xx = np.indices((size, size))
            re = 12.0 * resolution
            signal = Sersic2D(
                amplitude=120,
                r_eff=re,
                n=n,
                x_0=size / 2 + 0.3,
                y_0=size / 2 - 0.2,
                ellip=0.3,
                theta=0.4,
            )(xx, yy)
            yp, xp = np.indices((17, 17))
            psf = np.exp(-((xp - 8) ** 2 + (yp - 8) ** 2) / (2 * (1.2 * resolution) ** 2))
            psf /= psf.sum()
            data = (
                50.0
                + convolve(signal, psf)
                + np.random.default_rng(124).normal(0, noise, signal.shape)
            )
            sigma = np.full(signal.shape, noise)
            started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                metrics, seg = measure(data, psf, sigma)
            fitted_n, fitted_re = metrics["sersic_n"], metrics["sersic_rhalf"]
            n_error = abs(fitted_n / n - 1) if fitted_n is not None else None
            re_error = abs(fitted_re / re - 1) if fitted_re is not None else None
            passed = (
                n_error is not None and re_error is not None and n_error < 0.25 and re_error < 0.25
            )
            case = {
                "n": n,
                "r_eff": re,
                "shape": list(data.shape),
                "noise_sigma": noise,
                "metrics": metrics,
                "n_relative_error": n_error,
                "re_relative_error": re_error,
                "recovery_within_25_percent": passed,
                "seconds": time.perf_counter() - started,
                "warnings": sorted({str(item.message) for item in captured}),
            }
            controls.append(case)
            print(
                json.dumps(
                    {
                        k: case[k]
                        for k in (
                            "n",
                            "shape",
                            "noise_sigma",
                            "recovery_within_25_percent",
                            "seconds",
                        )
                    }
                ),
                flush=True,
            )
            if resolution == 1 and noise == 1 and n == 2.5:
                fits.HDUList(
                    [
                        fits.PrimaryHDU(),
                        fits.ImageHDU(data.astype(np.float32), name="SCI"),
                        fits.ImageHDU(sigma.astype(np.float32), name="ERR"),
                        fits.ImageHDU(psf.astype(np.float32), name="PSF"),
                    ]
                ).writeto(args.output / "scientific-input.fits", overwrite=True)
    real = None
    if args.galfit_example:
        with tarfile.open(args.galfit_example) as archive:
            raw = archive.extractfile("galfit-example/EXAMPLE/gal.fits").read()
            psf_raw = archive.extractfile("galfit-example/EXAMPLE/psf.fits").read()
        data = np.array(fits.getdata(io.BytesIO(raw)), dtype=float)
        psf = np.array(fits.getdata(io.BytesIO(psf_raw)), dtype=float)
        psf /= psf.sum()
        psf_note = (
            "Official example contains signed empirical PSF pixels; no automatic clipping "
            "was applied. Real-image comparison runs without a PSF."
        )
        if (psf < 0).any():
            psf = None
        metrics, seg = measure(data, psf)
        # Independent direct statmorph call checks adapter equivalence, not physical truth.
        import statmorph

        direct = statmorph.SourceMorphology(
            seg.data_sub,
            seg.tight_segmap_labels,
            seg.main_label,
            mask=seg.excluded_mask,
            weightmap=seg.noise_sigma,
            psf=psf,
        )
        comparison = {
            key: {
                "adapter": metrics[key],
                "direct": float(getattr(direct, key)),
                "matches": bool(np.isclose(metrics[key], getattr(direct, key), rtol=1e-6)),
            }
            for key in ("concentration", "asymmetry", "smoothness")
            if metrics[key] is not None
        }
        real = {
            "source": "https://users.obs.carnegiescience.edu/peng/work/galfit/galfit-ex.tar.gz",
            "sha256": hashlib.sha256(raw).hexdigest(),
            "shape": list(data.shape),
            "metrics": metrics,
            "adapter_equivalence": comparison,
            "psf_note": psf_note,
            "independent_physical_truth": False,
            "note": "feedme parameters are initial guesses, not published fitted reference values",
        }
        (args.output / "galfit-galaxy.fits").write_bytes(raw)
    report = {
        "synthetic_controls": controls,
        "real_fits": real,
        "all_synthetic_recovered": all(case["recovery_within_25_percent"] for case in controls),
        "limitations": [
            "Not a population-level scientific validation",
            "Independent pixel noise only",
            "Real example checks adapter equivalence, not agreement with published morphology",
        ],
    }
    (args.output / "science-validation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False)
    )
    if not report["all_synthetic_recovered"]:
        raise SystemExit("Recovery gate failed: inspect science-validation.json")


if __name__ == "__main__":
    main()
