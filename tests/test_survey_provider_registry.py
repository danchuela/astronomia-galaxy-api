from __future__ import annotations

from typing import Any

import pytest

from packages.galaxy_core.infrastructure.dss2_client import PLATE_BLUE
from packages.galaxy_core.infrastructure.irsa_finderchart_client import FINDERCHART_API
from packages.galaxy_core.infrastructure.survey_provider_registry import (
    resolve_image_url_for_survey,
)


def test_resolve_image_url_for_survey_uses_sdss_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_sdss_get_image_url(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "https://example.org/sdss.jpg"

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.survey_provider_registry.sdss_get_image_url",
        fake_sdss_get_image_url,
    )

    image_url = resolve_image_url_for_survey(
        survey="SDSS",
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=8.0,
    )

    assert image_url == "https://example.org/sdss.jpg"
    assert captured["ra_deg"] == 10.0
    assert captured["dec_deg"] == 20.0
    assert captured["size_arcmin"] == 8.0


def test_resolve_image_url_for_survey_uses_dss2_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_dss2_get_image_url(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "https://example.org/dss2-blue.jpg"

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.survey_provider_registry.dss2_get_image_url",
        fake_dss2_get_image_url,
    )

    image_url = resolve_image_url_for_survey(
        survey="DSS2-BLUE",
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=8.0,
    )

    assert image_url == "https://example.org/dss2-blue.jpg"
    assert captured["plate"] == PLATE_BLUE


def test_resolve_image_url_for_survey_uses_galex_strategy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_galex_get_image_url(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "https://example.org/galex.jpg"

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.survey_provider_registry.get_galex_jpeg_url",
        fake_galex_get_image_url,
    )

    image_url = resolve_image_url_for_survey(
        survey="GALEX",
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=6.0,
    )

    assert image_url == "https://example.org/galex.jpg"
    assert captured["ra_deg"] == 10.0
    assert captured["dec_deg"] == 20.0
    assert captured["galex_band"] == "NUV"


def test_resolve_image_url_for_survey_uses_2mass_strategy() -> None:
    image_url = resolve_image_url_for_survey(
        survey="2MASS-J",
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=9.0,
    )

    assert image_url.startswith(FINDERCHART_API)
    assert "survey=2mass" in image_url


def test_resolve_image_url_for_survey_falls_back_to_skyview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    def fake_skyview_get_image_url(**kwargs: Any) -> str:
        captured.update(kwargs)
        return "https://example.org/skyview.jpg"

    monkeypatch.setattr(
        "packages.galaxy_core.infrastructure.survey_provider_registry.skyview_get_image_url",
        fake_skyview_get_image_url,
    )

    image_url = resolve_image_url_for_survey(
        survey="WISE 3.4",
        ra_deg=10.0,
        dec_deg=20.0,
        size_arcmin=9.0,
    )

    assert image_url == "https://example.org/skyview.jpg"
    assert captured["survey"] == "WISE 3.4"
