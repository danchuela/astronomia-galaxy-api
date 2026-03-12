"""Target resolution and band-to-survey mapping."""

from __future__ import annotations

from dataclasses import dataclass

SurveyName = str

BAND_TO_SURVEY: dict[str, SurveyName] = {
    "visible": "SDSS",
    "optical": "SDSS",
    "infrared": "2MASS-J",
    "ir": "2MASS-J",
    "ultraviolet": "GALEX",
    "uv": "GALEX",
    "x-ray": "RASS",
    "xray": "RASS",
    "radio": "NVSS",
}


def get_capabilities_description() -> str:
    return (
        "Bandas disponibles: visible (óptico), infrarrojo, "
        "ultravioleta (uv), rayos X, radio. "
        "Catálogos/surveys que usa la aplicación: "
        "SDSS (visible, cielo norte), DSS/DSS2 (visible), "
        "DSS2-BLUE (visible azul), DSS2-IR (infrarrojo cercano), "
        "2MASS-J (infrarrojo), GALEX (ultravioleta), "
        "PanSTARRS, DECaLS, WISE, RASS (rayos X), "
        "XMM (rayos X), NVSS (radio). "
        "También se pueden usar otros surveys de SkyView o HiPS "
        "si se indica el nombre del catálogo. "
        "Para imágenes solo hay que indicar galaxia "
        "(nombre o coordenadas) y opcionalmente la banda; "
        "si no se indica banda, se usa visible por defecto."
    )


@dataclass(frozen=True)
class ResolvedTarget:
    ra_deg: float
    dec_deg: float
    name: str | None
    survey_used: str
    image_url: str
    size_arcmin: float
