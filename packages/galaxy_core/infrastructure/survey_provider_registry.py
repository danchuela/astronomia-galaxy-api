"""Survey provider registry."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from packages.galaxy_core.infrastructure.dss2_client import (
    get_image_url as dss2_get_image_url,
)
from packages.galaxy_core.infrastructure.dss2_client import (
    survey_to_plate as dss2_survey_to_plate,
)
from packages.galaxy_core.infrastructure.hips_client import (
    SURVEY_TO_HIPS,
)
from packages.galaxy_core.infrastructure.hips_client import (
    get_image_url as hips_get_image_url,
)
from packages.galaxy_core.infrastructure.irsa_finderchart_client import (
    FINDERCHART_API,
)
from packages.galaxy_core.infrastructure.mast_galex_client import get_galex_jpeg_url
from packages.galaxy_core.infrastructure.sdss_client import (
    get_image_url as sdss_get_image_url,
)
from packages.galaxy_core.infrastructure.skyview_client import (
    get_image_url as skyview_get_image_url,
)

ProviderResolver = Callable[[str, float, float, float], str | None]


@dataclass(frozen=True)
class SurveyProviderStrategy:
    name: str
    resolve_image_url: ProviderResolver


def _build_2mass_url(ra_deg: float, dec_deg: float, size_arcmin: float) -> str:
    return (
        f"{FINDERCHART_API}"
        f"?mode=getImage&file_type=png&survey=2mass"
        f"&subsetsize={float(size_arcmin)}"
        f"&RA={ra_deg}&DEC={dec_deg}"
        f"&reproject=true&marker=false&grid=false"
    )


def _resolve_sdss_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    if survey.upper() != "SDSS":
        return None
    return sdss_get_image_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        size_arcmin=size_arcmin,
    )


def _resolve_dss2_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    plate = dss2_survey_to_plate(survey)
    if plate is None:
        return None
    return dss2_get_image_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        size_arcmin=size_arcmin,
        plate=plate,
    )


def _resolve_galex_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    if survey.upper() != "GALEX":
        return None
    return get_galex_jpeg_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        radius_deg=max(0.05, float(size_arcmin) / 60.0),
        galex_band="NUV",
    )


def _resolve_2mass_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    if survey.upper() != "2MASS-J":
        return None
    return _build_2mass_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        size_arcmin=size_arcmin,
    )


def _resolve_hips_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    key = survey if survey in SURVEY_TO_HIPS else survey.upper()
    if key not in SURVEY_TO_HIPS:
        return None
    return hips_get_image_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        survey=key,
        size_arcmin=size_arcmin,
    )


def _resolve_skyview_url(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str | None:
    return skyview_get_image_url(
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        survey=survey,
        size_arcmin=size_arcmin,
    )


SURVEY_PROVIDER_REGISTRY: tuple[SurveyProviderStrategy, ...] = (
    SurveyProviderStrategy(name="sdss", resolve_image_url=_resolve_sdss_url),
    SurveyProviderStrategy(name="dss2", resolve_image_url=_resolve_dss2_url),
    SurveyProviderStrategy(name="galex", resolve_image_url=_resolve_galex_url),
    SurveyProviderStrategy(name="2mass", resolve_image_url=_resolve_2mass_url),
    SurveyProviderStrategy(name="hips", resolve_image_url=_resolve_hips_url),
    SurveyProviderStrategy(
        name="skyview-fallback",
        resolve_image_url=_resolve_skyview_url,
    ),
)


def resolve_image_url_for_survey(
    survey: str,
    ra_deg: float,
    dec_deg: float,
    size_arcmin: float,
) -> str:
    last_error: Exception | None = None
    for strategy in SURVEY_PROVIDER_REGISTRY:
        try:
            image_url = strategy.resolve_image_url(
                survey,
                ra_deg,
                dec_deg,
                size_arcmin,
            )
            if image_url is not None:
                return image_url
        except Exception as exc:
            last_error = exc
            continue
    if last_error is not None:
        raise last_error
    raise ValueError(f"No provider available for survey {survey!r}")
