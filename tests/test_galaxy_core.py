from __future__ import annotations

import numpy as np

from packages.galaxy_core.analyzer import BasicGalaxyAnalyzer, create_synthetic_image


def test_segment_galaxy_returns_binary_mask() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image()

    result = analyzer.segment_galaxy(image)

    assert result.mask.shape == image.shape
    assert set(np.unique(result.mask)).issubset({0, 1})
    assert result.metadata["algorithm"] in ("sep", "quantile_fallback")


def test_segment_galaxy_sep_populates_extra_fields() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image()

    result = analyzer.segment_galaxy(image)

    if result.metadata["algorithm"] == "sep":
        assert result.data_sub is not None
        assert result.segmap_labels is not None
        assert result.segmap_labels.shape == image.shape


def test_measure_basic_returns_expected_keys() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image((64, 64))
    segmentation = analyzer.segment_galaxy(image)

    measurements = analyzer.measure_basic(image, segmentation)

    base_keys = {"area_pixels", "centroid_x", "centroid_y", "ellipticity", "mean_intensity"}
    assert base_keys.issubset(measurements.keys())
    assert measurements["area_pixels"] > 0
    assert "radial_profile" in measurements
    assert "analysis_reliable" in measurements


def test_measure_basic_statmorph_keys_when_sep() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image((64, 64))
    segmentation = analyzer.segment_galaxy(image)

    if segmentation.metadata["algorithm"] != "sep":
        return  # statmorph sólo se ejecuta con SEP

    measurements = analyzer.measure_basic(image, segmentation)

    statmorph_keys = {"concentration", "asymmetry", "flag"}
    assert statmorph_keys.issubset(measurements.keys())


def test_morphology_summary_is_non_empty() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image((64, 64))
    segmentation = analyzer.segment_galaxy(image)
    measurements = analyzer.measure_basic(image, segmentation)

    summary = analyzer.morphology_summary(measurements)

    assert len(summary) > 0
    assert "Detected structure area" in summary


def test_radial_profile_structure() -> None:
    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image((64, 64))
    segmentation = analyzer.segment_galaxy(image)
    measurements = analyzer.measure_basic(image, segmentation)

    rp = measurements["radial_profile"]
    assert "radii_px" in rp
    assert "mean_flux" in rp
    assert len(rp["radii_px"]) == len(rp["mean_flux"])
    assert len(rp["radii_px"]) > 0


def test_isophotes_returns_table_and_png() -> None:
    from packages.galaxy_core.application.isophotes import (
        compute_isophotes,
        format_isophotes_summary,
    )

    analyzer = BasicGalaxyAnalyzer()
    image = create_synthetic_image((128, 128))
    segmentation = analyzer.segment_galaxy(image)
    measurements = analyzer.measure_basic(image, segmentation)

    iso_table, png_bytes = compute_isophotes(image, segmentation.mask, measurements)

    # La función siempre devuelve bytes PNG aunque el ajuste no converja
    assert isinstance(png_bytes, bytes)
    assert len(png_bytes) > 0

    summary = format_isophotes_summary(iso_table, "test_galaxy")
    assert isinstance(summary, str)
    assert len(summary) > 0
