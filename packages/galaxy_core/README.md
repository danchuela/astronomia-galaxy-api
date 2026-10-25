# Galaxy Image Analysis

Pure numerical layer, without an LLM or HTTP dependencies in the analysis algorithms.

## Input and Scope

The current application analyzes rendered survey images or the Aladin image plane.
RGB inputs are converted to Pillow luminance; native grayscale precision is retained.
PNG capture avoids an additional lossy encoding, but does not make the survey rendering
linear, calibrated, or single-band scientific data. Results are exploratory measurements
of the supplied pixels, not calibrated photometry or validated galaxy classifications.

The original bytes remain immutable. Segmentation annotations are separate plots.
Request results include the input SHA-256, image dimensions, capture geometry, software
versions, module status, and input limitations. Different surveys, stretches, resolutions,
fields, and segmentations can yield different measurements of the same galaxy.

## Segmentation and Background

SEP supplies a spatial background and a background RMS (sigma) map. The background is
subtracted before all intensity measurements. Invalid pixels are excluded. The source
with the highest `area / (distance_to_image_center_squared + 1)` is selected, not
necessarily the named galaxy; multiple sources and boundary truncation are recorded.
Automatic identity association or interactive target-mask correction is not implemented.

The complete SEP label map is retained to exclude detected neighbors. The primary mask
is grown by up to three pixels above 1.5 times background sigma, without overwriting
other source labels. Statmorph receives the tight label map. No detection is an expected
clarification outcome: no percentile-based replacement source is manufactured.

The background mesh is a heuristic, not a jointly fitted sky model. Extended sources
that dominate the frame require a larger field. Neighbor masks do not remove undetected
sources, blended flux, or all PSF wings.

## CAS and Geometry

Statmorph computes concentration (`5 log10(r80/r20)`), background-corrected rotational
asymmetry, and its implemented smoothness convention. These are not interchangeable
with every CAS implementation. Negative background-corrected A/S are retained;
nonfinite values and statmorph's `-99` failure sentinel become null.

Statmorph's `weightmap` is a **standard-deviation image**, not inverse variance.
The current map estimates background noise only: it lacks calibrated source Poisson
noise and resampling covariance. `flag >= 2` suppresses CAS; `flag == 1` is reported as
suspect. A quality flag of zero is not proof of scientific accuracy. Published merger
or Hubble-type boundaries are not applied to arbitrary rendered images.

Basic centroid and ellipticity are positive-intensity-weighted second moments of the
background-subtracted primary region. The clipped positive weights are an explicit
geometry convention, not an unbiased low-S/N estimator. Flux and mean intensity retain
signed sky-subtracted values. `asymmetry_center_x/y` are separate from the flux centroid.

## Single-Component Sersic Fit

Statmorph fits one two-dimensional Sersic component. No artificial universal PSF is
assumed. A supplied numerical PSF is normalized and used in convolution; otherwise the
fit is explicitly not PSF-corrected. Rendered-viewer analysis currently has no measured
PSF. The fit is not a bulge/disk decomposition, and an `n < 2` versus `n > 2` distinction
is not guaranteed to classify the input accurately.

Sersic parameters are withheld when `flag_sersic != 0` or the global fit is invalid.
The diagnostic uses the **actual fitted model and its fitting stamp**, including its
PSF when supplied: data, model, data-minus-model, and residual divided by background
sigma. Data/model share one intensity normalization; residual scales are centered on
zero. Contaminants and invalid pixels are hidden. The pinned statmorph adapter accesses
its fitted callable; dependency upgrades must run the recovery and residual tests.
Parameter covariance, posterior intervals, PSF uncertainty, and sky uncertainty are
not currently estimated. Residual structure is not automatically identified as arms,
star formation, or interactions.

## Radial Profile

Mean signed background-subtracted intensity is measured in nonoverlapping elliptical
annuli using the moment centroid, ellipticity, and position angle. The last radius comes
from the primary mask. Neighbor and invalid pixels are excluded; unlabeled background
within the annuli is retained. Output includes radii in pixels, mean intensity, valid
pixel counts, and `sqrt(sum(sigma_i**2)) / N` when sigma is available.

These errors assume independent background noise only. They are not total photometric
errors, parameter confidence intervals, or calibrated surface brightness in mag/arcsec2.
No PSF deconvolution is performed. `n_bins` must be between 4 and 100.

## Isophotes

Photutils fits ellipses on the signed background-subtracted image with invalid and
neighbor pixels masked. A valid result requires at least three valid isophotes. The
table preserves intensity, ellipticity, position angle, center, available fitting
errors, and stopping codes without rounding away precision. Formal fitting errors do
not include calibration, PSF, sky, or correlated-noise systematics.

The separate visualization draws contours on a smoothed display image. These contours
are **not** the fitted ellipses. Failed ellipse fitting returns an empty table with an
unavailable module status; the segmentation boundary is never substituted for an
isointensity contour. Lack of convergence is not itself evidence of an interaction.

## Quality and Validation

Module availability and completeness are code-owned. Missing/failed modules produce
partial or unavailable results, not an LLM assertion that all measurements completed.
Catalog morphology is labeled separately from image measurements. The UI has one
collapsible quality section instead of repeating caveats on every plot.

Tests cover blank fields, sigma semantics, neighbor exclusion, negative values,
invalid flags, known moment geometry, background-subtracted annuli, actual-model
residuals, and recovery of controlled PSF-convolved synthetic Sersic sources. They do
not establish observational accuracy across heterogeneous surveys. A validated
scientific mode still needs native linear science pixels, WCS/band/PSF/error metadata,
calibration, systematic-error assessment, and an external reference benchmark.

References: [statmorph API](https://statmorph.readthedocs.io/en/latest/api.html),
[statmorph implementation](https://statmorph.readthedocs.io/en/latest/_modules/statmorph/statmorph.html),
[Photutils isophotes](https://photutils.readthedocs.io/en/stable/user_guide/isophote.html).
