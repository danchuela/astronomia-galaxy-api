# Image analysis contracts

## Own images

PNG, JPEG, TIFF and standard uncompressed 2D FITS are accepted, up to 8 MiB
and 4,194,304 pixels. FITS intensities retain BSCALE/BZERO scaling, signed values
and native floating precision; the display preview is never measured. Uploads
do not acquire catalogue metadata or coordinates automatically.

`POST /images/inspect` validates an `UploadedImage` and returns a bounded display
preview, dimensions, detected source positions and FITS image-HDU headers.
It makes no model calls and saves no artifacts. It uses the same auth and body
limits as analysis. File bytes must not be put in application logs or localStorage.

`source_point: {x, y}` uses normalized **image-array** coordinates, from the
upper-left preview corner, not sky coordinates or FITS one-based coordinates.
Selecting background never substitutes the automatic central source. `region`
has normalized `left`, `top`, `right`, `bottom` edges and excludes pixels outside
it; it does not resize the image. Returned centroids remain in original-image
pixels. Background estimation still uses the whole science image. A source
touching either the image edge or the selected region is flagged.

## Auxiliary FITS planes

Select `hdu_index` for science, `psf_hdu_index` for a PSF and `error_hdu_index`
for an error map **inside the same file**. None are inferred from an arbitrary
HDU number. The error map must match science dimensions and intensity units;
`error_kind` explicitly selects sigma, variance or inverse variance. Invalid,
zero or negative errors are excluded, not replaced with invented measurements.

A supplied PSF must be centered to within one pixel, finite, nonnegative, odd-sized,
at most 129 pixels per side and at the science image's pixel scale and band.
It is normalized and passed to statmorph's convolved Sersic model. Signed empirical
PSFs require preparation outside this interface; they are not silently clipped.
Matching band, scale, alignment and calibration remain the user's responsibility.

## Noise stability

`uncertainty_samples: 16` opts into reproducible independent Gaussian perturbations
on images at most 512 by 512 pixels (default off, seed 42). Segmentation, background,
sigma and PSF remain fixed. At least 12 valid measurements per parameter are needed
to report sample standard deviation and 16th/84th percentiles. Failed draws are
counted, never imputed. These are **conditional noise stability estimates**, not
confidence intervals or total parameter uncertainties. Correlated noise, PSF,
background, calibration and source-selection systematics are excluded.

## Capacity

One scientific POST is admitted at a time per API process, before its body is read.
Concurrent excess requests return HTTP 503 with `Retry-After: 3`; health and artifact
downloads remain available. The BFF reports that another analysis is running, not
that the module is broken. Production runs one Uvicorn worker. Increase workers
only after measuring memory per process. This bounds admission, not execution time.

Detection uses a bounded SEP active-pixel buffer (up to 1,048,576 pixels).
CAS/Sersic/isophote workloads whose standard 2.5x source stamp would exceed
589,824 pixels are explicitly unavailable. Segmentation and radial measurements
can still be returned. No pixels are downsampled to fit this budget; use a native
cutout with sufficient surrounding sky. The 4 MP file limit is not a promise
that every extended source can be fitted in this small VPS.

## Validation

`python scripts/validate_science.py --output <folder> --galfit-example <archive>`
checks recovery on nine known single-Sersic controls (indices 1, 2.5, 4; two pixel
resolutions; two noise levels) and checks adapter equivalence against a direct
statmorph call on the official GALFIT real-image example. The feedme values are
initial guesses, not published truth. This is a reproducible baseline, **not** a
population-level accuracy study or equivalence to GALFIT/multicomponent fitting.

References:
- https://statmorph.readthedocs.io/en/latest/api.html
- https://statmorph.readthedocs.io/en/latest/notebooks/tutorial.html
- https://users.obs.carnegiescience.edu/peng/work/galfit/galfit.html
