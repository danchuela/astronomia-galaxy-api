# galaxy_core — Morphological Analysis

Pure data-science layer. No LLM, no FastAPI.

## Analysis modules

### CAS — Concentration, Asymmetry, Smoothness

Based on Conselice (2003, ApJS 147, 1). Computed via **statmorph** on background-subtracted images using the tight segmap (raw SEP output) to avoid diluting metrics with diffuse tidal material.

| Parameter | Formula | Typical range | What it measures |
|-----------|---------|---------------|-----------------|
| **C** | 5 log(r80 / r20) | 1–5 | Central light concentration. High C → dominant bulge or compact core. |
| **A** | Σ\|I − I₁₈₀\| / Σ\|I\| | 0–1 | Rotational asymmetry. A > 0.35 (Conselice 2003 merger threshold). |
| **S** | Σ\|I − I_smooth\| / Σ\|I\| | 0–0.5 | High-frequency structure: star-forming clumps, dust lanes. Near 0 or negative for smooth ellipticals. |

M101 (spiral, S_AB): C=2.24, A=0.28, S=−0.06 — M87 (elliptical, AGN): C=2.21, A=−0.05, S=−0.05 — NGC 4038 (merger): C=1.79, A=0.46, S=0.02

![CAS indices — M101 spiral](../../docs/assets/cas-m101.png)
*M101 (spiral). A=0.28 indicates moderate asymmetry from spiral arms.*

![CAS indices — NGC 4038 merger](../../docs/assets/cas-ngc4038.png)
*NGC 4038/4039 (Antennae merger). A=0.46 exceeds the 0.35 merger threshold.*

**Additional statmorph fields** returned alongside CAS:

| Field | Description |
|-------|-------------|
| `rpetro_circ` | Circular Petrosian radius (px): the radius where the annular mean surface brightness drops to 20% of the interior mean. Standard galaxy size estimator. |
| `r20` / `r80` | Radii enclosing 20% and 80% of the Petrosian flux. Used to compute C. |
| `ellipticity_asymmetry` | Ellipticity of the best-fit ellipse to the asymmetry distribution. |
| `elongation_asymmetry` | Semi-major / semi-minor axis ratio of the asymmetry ellipse. |
| `orientation_asymmetry` | Position angle of the asymmetry ellipse (rad). Used as initial PA for isophote fitting. |
| `centroid_x` / `centroid_y` | Position of the asymmetry minimum (px). Differs from flux centroid in mergers or lopsided galaxies. |
| `sn_per_pixel` | Median S/N per pixel inside the segmap. Values below ~2.5 indicate unreliable CAS measurements. |
| `flag` | statmorph quality flag: 0 = clean, 1 = minor issues, 2 = unreliable. |
| `flag_sersic` | 0 = Sérsic fit converged; non-zero = fit failed, Sérsic parameters not reported. |

**Ground-based caveat:** PSF smearing lowers C and S artificially. Do not compare values directly against HST-calibrated classification boundaries.

---

### Sérsic profile fit

Based on Sérsic (1963). Fitted by statmorph with a 2D model convolved with a synthetic Gaussian PSF (FWHM = 2 px). Skipped when `flag_sersic > 0`.

| Parameter | Description | Interpretation |
|-----------|-------------|----------------|
| `sersic_n` | Sérsic index | n ≈ 1: exponential disk. n ≈ 4: de Vaucouleurs elliptical. n > 4: compact nucleus. |
| `sersic_rhalf` | Half-light radius (px) | Radius enclosing 50% of the fitted flux. |
| `sersic_ellip` | Ellipticity (1 − b/a) | 0 = circular. ~0.3–0.6 = inclined disk. > 0.7 = edge-on. |
| `sersic_theta` | Position angle (rad) | Orientation of the major axis. |

**Ground-based caveat:** PSF correction uses a synthetic Gaussian, not a measured PSF. The n < 2 / n > 2 distinction (disk vs. bulge-dominated) is reliable; fine-grained n values are not.

**Three-panel plot** produced when the fit converges:

| Panel | Content |
|-------|---------|
| **Data (background-sub)** | Input image after SEP background subtraction — the pixel data statmorph receives. |
| **Sérsic Model** | Fitted 2D Sérsic function on the same grid. Smooth and symmetric by definition; no arms or bars. |
| **Residual (data − model)** | Pixel-by-pixel difference. Red = galaxy exceeds model (spiral arms, HII regions). Blue = model overestimates (outer disk). For spirals this maps non-axisymmetric structure. |

![Sérsic 2D fit — M101](../../docs/assets/sersic-m101.png)
*M101 (spiral). n=0.56 indicates a disk-dominated profile. The residual clearly shows the spiral arms and HII knots not captured by the smooth model.*

---

### Radial brightness profile

Computed directly on the image (not via statmorph). Mean flux in N concentric annuli from the centroid to the segmentation mask edge. `n_bins` is configurable (default 25).

Output: `{"radii_px": [...], "mean_flux": [...]}`.

| Profile shape | Implication |
|---------------|-------------|
| Smooth exponential decline | Disk-dominated |
| Steep central peak + shallow outer slope | Bulge + disk composite |
| Flat center + sharp drop | Bar, ring, or saturated nucleus |
| Irregular / multiple bumps | Disturbed morphology or interaction |

No PSF correction. The innermost annulus (~2–3 px) is PSF-smeared and should not be interpreted as the true nuclear brightness.

![Radial brightness profile — M101](../../docs/assets/radial-profile-m101.png)
*M101 (spiral). Smooth exponential decline consistent with a disk-dominated profile.*

---

### Isophotes

Elliptical isophote fitting via **photutils** `Ellipse`. Fits isophotes at increasing semi-major axes and returns a table with one row per isophote.

| Field | Description |
|-------|-------------|
| `sma` | Semi-major axis (px) |
| `intens` | Mean intensity along the isophote |
| `ellip` | Ellipticity (1 − b/a) |
| `pa_deg` | Position angle of the major axis (degrees) |
| `x0` / `y0` | Isophote center (px) — may drift for lopsided galaxies |

| Pattern | Implication |
|---------|-------------|
| Ellipticity increasing with radius | Disk becomes dominant outside the bulge |
| Constant ellipticity and PA | Smooth symmetric disk |
| PA twist with radius | Bar, warped disk, or triaxial bulge |
| Non-convergence / large scatter | Irregular morphology or tidal distortion |

**Visualization:** isointensity contours are drawn on the image using skimage `find_contours`. If photutils fails to converge, a single fallback contour at the segmap boundary is drawn instead.

UV (GALEX) and radio surveys produce irregular emission that degrades ellipse fitting; optical surveys (SDSS, DSS2, PanSTARRS) give the most reliable results.

![Isophotes — M101 spiral](../../docs/assets/isophotes-m101.png)
*M101 (spiral). Irregular outer contours follow the spiral arms; inner isophotes are more regular.*

![Isophotes — M87 elliptical](../../docs/assets/isophotes-m87.png)
*M87 (elliptical). Concentric, nearly circular isophotes — characteristic of a relaxed elliptical.*

---

## Segmentation

Two segmaps are produced from a single SEP run:

- **tight_segmap** — raw SEP output. Used as statmorph input to avoid diluting CAS metrics with diffuse outer flux.
- **grown_segmap** — tight + 3 px dilation at 1.5σ. Used only for visual contour overlays.

The main source is the largest detected object closest to the image center. If SEP finds no sources, a quantile-threshold fallback mask is used (`segmentation_metadata.algorithm = "quantile_fallback"`).

The annotated image shows the grown segmap boundary in green and the asymmetry centroid in orange.

![Segmentation — M87 elliptical](../../docs/assets/segmentation-m87.png)
*M87 (elliptical). Single clean contour around a symmetric source.*

![Segmentation — NGC 4038 merger](../../docs/assets/segmentation-ngc4038.png)
*NGC 4038/4039 (Antennae). The segmap captures both nuclei and the tidal tails.*
