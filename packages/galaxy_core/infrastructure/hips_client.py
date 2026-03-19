"""CDS HiPS image cutout URL builder (HiPS2FITS API)."""

from __future__ import annotations

import urllib.parse

HIPS2FITS_BASE = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
DEFAULT_PIXELS = 512

SURVEY_TO_HIPS: dict[str, str] = {
    "SDSS": "CDS/P/SDSS9/color",
    "DSS": "CDS/P/DSS2/color",
    "DSS2": "CDS/P/DSS2/color",
    "DSS2-RED": "CDS/P/DSS2/red",
    "DSS2-BLUE": "CDS/P/DSS2/blue",
    "DSS2-IR": "CDS/P/DSS2/red",
    "PANSTARRS": "CDS/P/PanSTARRS/DR1/color-i-r-g",
    "DECALS": "CDS/P/DECaLS/DR5/color",
    "2MASS-J": "CDS/P/2MASS/J",
    "2MASS": "CDS/P/2MASS/color",
    "WISE": "CDS/P/allWISE/color",
    "WISE-W1": "CDS/P/allWISE/W1",
    "ALLWISE": "CDS/P/allWISE/color",
    "GALEX": "CDS/P/GALEXGR6_7/NUV",
    "GALEX-NUV": "CDS/P/GALEXGR6_7/NUV",
    "GALEX-FUV": "CDS/P/GALEXGR6_7/FUV",
    "GALEX-COLOR": "CDS/P/GALEXGR6_7/color",
    "RASS": "CDS/P/RASS",
    "XMM": "xcatdb/P/XMM/PN/color",
    "NVSS": "CDS/P/NVSS",
}


def get_image_url_from_hips_id(
    ra_deg: float,
    dec_deg: float,
    hips_id: str,
    size_arcmin: float = 10.0,
    pixels: int = DEFAULT_PIXELS,
) -> str:
    """Build a hips2fits URL from a HiPS ID."""
    fov_deg = size_arcmin / 60.0
    params = {
        "hips": hips_id,
        "width": pixels,
        "height": pixels,
        "projection": "SIN",
        "fov": round(fov_deg, 6),
        "ra": round(ra_deg, 6),
        "dec": round(dec_deg, 6),
        "format": "jpg",
        "coordsys": "icrs",
    }
    return f"{HIPS2FITS_BASE}?{urllib.parse.urlencode(params)}"


def survey_to_hips_id(survey: str) -> str | None:
    """Return the HiPS ID for a given survey name, or None if not mapped."""
    return SURVEY_TO_HIPS.get(survey) or SURVEY_TO_HIPS.get(survey.upper())


def get_image_url(
    ra_deg: float,
    dec_deg: float,
    survey: str,
    size_arcmin: float = 10.0,
    pixels: int = DEFAULT_PIXELS,
) -> str:
    hips_id = SURVEY_TO_HIPS.get(survey) or SURVEY_TO_HIPS.get(survey.upper())
    if not hips_id:
        raise ValueError(
            f"HiPS does not support survey {survey!r}. Supported: {list(SURVEY_TO_HIPS.keys())}"
        )
    fov_deg = size_arcmin / 60.0
    params = {
        "hips": hips_id,
        "width": pixels,
        "height": pixels,
        "projection": "SIN",
        "fov": round(fov_deg, 6),
        "ra": round(ra_deg, 6),
        "dec": round(dec_deg, 6),
        "format": "jpg",
        "coordsys": "icrs",
    }
    return f"{HIPS2FITS_BASE}?{urllib.parse.urlencode(params)}"
