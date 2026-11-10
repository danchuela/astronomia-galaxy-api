"""Viewer products hosted outside the CDS keep their survey name and band in reports."""

from __future__ import annotations

import pytest

from packages.galaxy_agent.agent_tools import _band_from_survey, _survey_display_label
from packages.galaxy_core.infrastructure.hips_client import survey_to_hips_id


@pytest.mark.parametrize(
    ("hips_id", "label", "band"),
    [
        ("CDS/P/GALEXGR6/AIS/FUV", "GALEX-FUV", "ultravioleta"),
        ("CDS/P/GALEXGR6/AIS/NUV", "GALEX-NUV", "ultravioleta"),
        ("ESDC/P/XMM/EPIC-RGB", "XMM", "rayos X"),
        ("erosita/dr1/rate/rgb", "eROSITA", "rayos X"),
    ],
)
def test_new_viewer_products_have_label_and_band(hips_id: str, label: str, band: str) -> None:
    assert _survey_display_label(hips_id) == label
    assert _band_from_survey(hips_id) == band


def test_galex_opens_the_product_also_hosted_by_irsa() -> None:
    assert survey_to_hips_id("GALEX") == "CDS/P/GALEXGR6/AIS/NUV"
    assert survey_to_hips_id("GALEX-FUV") == "CDS/P/GALEXGR6/AIS/FUV"
