from __future__ import annotations

import pytest

from packages.galaxy_core.infrastructure.hips_client import (
    HIPS2FITS_BASE,
    get_image_url,
    survey_to_hips_id,
)
from packages.galaxy_core.infrastructure.survey_provider_registry import (
    resolve_image_url_for_survey,
)


class TestSurveyToHipsId:
    def test_known_survey(self) -> None:
        assert survey_to_hips_id("SDSS") == "CDS/P/SDSS9/color"

    def test_case_insensitive(self) -> None:
        assert survey_to_hips_id("panstarrs") == "CDS/P/PanSTARRS/DR1/color-i-r-g"
        assert survey_to_hips_id("wise") == "CDS/P/allWISE/color"

    def test_unknown_returns_none(self) -> None:
        assert survey_to_hips_id("FOOBAR_SURVEY") is None


class TestHipsGetImageUrl:
    def test_known_survey_builds_url(self) -> None:
        url = get_image_url(ra_deg=10.0, dec_deg=20.0, survey="PanSTARRS")
        assert HIPS2FITS_BASE in url
        assert "PanSTARRS" in url

    def test_case_insensitive_lookup(self) -> None:
        url = get_image_url(ra_deg=10.0, dec_deg=20.0, survey="wise")
        assert "allWISE" in url

    def test_unknown_survey_raises(self) -> None:
        with pytest.raises(ValueError, match="does not support"):
            get_image_url(ra_deg=10.0, dec_deg=20.0, survey="UNKNOWN_XYZ")


class TestHipsInRegistry:
    def test_panstarrs_resolves_via_hips(self) -> None:
        url = resolve_image_url_for_survey(
            survey="PanSTARRS",
            ra_deg=10.0,
            dec_deg=20.0,
            size_arcmin=8.0,
        )
        assert "alasky.cds.unistra.fr" in url

    def test_wise_resolves_via_hips(self) -> None:
        url = resolve_image_url_for_survey(
            survey="WISE",
            ra_deg=10.0,
            dec_deg=20.0,
            size_arcmin=8.0,
        )
        assert "alasky.cds.unistra.fr" in url

    def test_nvss_resolves_via_hips(self) -> None:
        url = resolve_image_url_for_survey(
            survey="NVSS",
            ra_deg=10.0,
            dec_deg=20.0,
            size_arcmin=8.0,
        )
        assert "alasky.cds.unistra.fr" in url

    def test_rass_resolves_via_hips(self) -> None:
        url = resolve_image_url_for_survey(
            survey="RASS",
            ra_deg=10.0,
            dec_deg=20.0,
            size_arcmin=8.0,
        )
        assert "alasky.cds.unistra.fr" in url
