"""Resolve target and fetch image URL from catalog."""

from __future__ import annotations

from packages.galaxy_core.domain.imaging import BAND_TO_SURVEY, ResolvedTarget
from packages.galaxy_core.infrastructure.sesame_client import resolve as sesame_resolve
from packages.galaxy_core.infrastructure.survey_provider_registry import (
    resolve_image_url_for_survey,
)


def _resolve_target_coordinates(
    name: str | None,
    ra_deg: float | None,
    dec_deg: float | None,
) -> tuple[float, float, str | None]:
    if name is not None and name.strip() != "":
        if ra_deg is not None or dec_deg is not None:
            raise ValueError("Provide either name or (ra_deg, dec_deg), not both.")
        resolved_name = name.strip()
        resolved_ra, resolved_dec = sesame_resolve(resolved_name)
        return resolved_ra, resolved_dec, resolved_name
    if ra_deg is not None and dec_deg is not None:
        return ra_deg, dec_deg, None
    raise ValueError("Provide either name or (ra_deg, dec_deg).")


def _resolve_survey_name(band: str | None, catalog: str | None) -> str:
    if catalog is not None and catalog.strip() != "":
        return catalog.strip()
    if band is None or band.strip() == "":
        raise ValueError("Provide either band or catalog.")

    mapped = BAND_TO_SURVEY.get(band.strip().lower())
    if mapped is None:
        raise ValueError(f"Unknown band '{band}'. Use one of: {list(BAND_TO_SURVEY.keys())}")
    return mapped


def resolve_and_fetch(
    name: str | None = None,
    ra_deg: float | None = None,
    dec_deg: float | None = None,
    band: str | None = None,
    catalog: str | None = None,
    size_arcmin: float = 10.0,
) -> ResolvedTarget:
    resolved_ra, resolved_dec, resolved_name = _resolve_target_coordinates(
        name=name,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
    )
    survey_str = _resolve_survey_name(band=band, catalog=catalog)
    image_url = resolve_image_url_for_survey(
        survey=survey_str,
        ra_deg=resolved_ra,
        dec_deg=resolved_dec,
        size_arcmin=size_arcmin,
    )

    return ResolvedTarget(
        ra_deg=resolved_ra,
        dec_deg=resolved_dec,
        name=resolved_name,
        survey_used=survey_str,
        image_url=image_url,
        size_arcmin=size_arcmin,
    )
