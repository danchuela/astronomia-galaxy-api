"""User-facing texts of the Galaxy API, in Spanish and English.

The answer language comes from the BFF (`answer_language` in the request, see
astronomia-ui-backend app/lang.py); requests without it are answered in Spanish.
Every function takes ``lang`` ("es" or "en") and defaults to Spanish, so the
Spanish texts are unchanged.
"""

from __future__ import annotations

from typing import Any, Literal


def image_capacity_message(lang: str = "es") -> str:
    return (
        "La fuente requiere más recursos de los disponibles para este análisis. "
        "Sube un recorte a resolución original con margen para el fondo. "
        "No he reducido la imagen ni rellenado las medidas no calculadas."
        if lang == "es"
        else "This source exceeds the resource budget for this analysis. "
        "Upload a native-resolution cutout with enough surrounding sky. "
        "I have not resized the image or filled in unmeasured values."
    )


Lang = Literal["es", "en"]

TARGET_REQUIRED_MESSAGE = (
    "Indica de qué galaxia quieres la imagen o el análisis "
    "(por nombre, p. ej. M104, o por coordenadas). "
    "Puedes añadir la banda: visible, infrarrojo o ultravioleta."
)
TARGET_REQUIRED_MESSAGE_EN = (
    "Tell me which galaxy you want the image or the analysis of "
    "(by name, e.g. M104, or by coordinates). "
    "You can add the band: visible, infrared or ultraviolet."
)

ERROR_ANALYSIS_FAILED_MESSAGE = "No se pudo completar el análisis. Inténtalo de nuevo."
ERROR_ANALYSIS_FAILED_MESSAGE_EN = "The analysis could not be completed. Please try again."
ERROR_IMAGE_FETCH_FAILED_MESSAGE = (
    "No se pudo obtener la imagen para esa banda en este momento. "
    "Prueba otra banda o inténtalo de nuevo."
)
ERROR_IMAGE_FETCH_FAILED_MESSAGE_EN = (
    "The image for that band could not be retrieved right now. Try another band or try again."
)

# Progress messages shown while the agent calls each tool.
TOOL_STATUS: dict[str, dict[str, str]] = {
    "es": {
        "resolve_and_fetch_image": "Resolviendo objetivo…",
        "segment_image": "Segmentando imagen…",
        "analyze_galaxy": "Calculando análisis morfológico…",
        "run_isophotes": "Ajustando isofotas elípticas…",
        "enrich_metadata": "Consultando catálogos externos…",
        "generate_final_report": "Generando resumen…",
    },
    "en": {
        "resolve_and_fetch_image": "Resolving target…",
        "segment_image": "Segmenting image…",
        "analyze_galaxy": "Computing morphological analysis…",
        "run_isophotes": "Fitting elliptical isophotes…",
        "enrich_metadata": "Querying external catalogues…",
        "generate_final_report": "Writing summary…",
    },
}


def lang_of(request: Any) -> Lang:
    """Answer language of a request: "en" only when the BFF asked for English."""
    return "en" if getattr(request, "answer_language", None) == "en" else "es"


def target_required_message(lang: str = "es") -> str:
    return TARGET_REQUIRED_MESSAGE_EN if lang == "en" else TARGET_REQUIRED_MESSAGE


def analysis_failed_message(lang: str = "es") -> str:
    return ERROR_ANALYSIS_FAILED_MESSAGE_EN if lang == "en" else ERROR_ANALYSIS_FAILED_MESSAGE


def image_fetch_failed_message(lang: str = "es") -> str:
    return ERROR_IMAGE_FETCH_FAILED_MESSAGE_EN if lang == "en" else ERROR_IMAGE_FETCH_FAILED_MESSAGE


def target_not_found_message(name: str, lang: str = "es") -> str:
    if lang == "en":
        return (
            f'I could not find an object named "{name}". '
            "Check the designation or give me another name or its coordinates (RA/Dec)."
        )
    return (
        f'No he encontrado un objeto con el nombre "{name}". '
        "Comprueba la designación o indícame otro nombre o sus coordenadas (RA/Dec)."
    )


def the_object(lang: str = "es") -> str:
    return "the object" if lang == "en" else "el objeto"


def viewer_ready_message(label: str, lang: str = "es", list_analyses: bool = True) -> str:
    """Answer when the object is only placed in the viewer (task "resolve").

    The BFF turns the list of analyses into buttons: keep these sentences in sync
    with astronomia-ui-backend app/analysis_actions.py (VIEWER_READY_MARKER*).
    """
    if lang == "en":
        text = (
            f"Here is {label}. "
            "Adjust the framing, zoom and band in the viewer as you like. "
            "When you are ready, tell me what you want to analyse"
        )
        tail = (
            ": full morphology, segmentation and isophotes, "
            "basic photometry, or brightness profile."
        )
    else:
        text = (
            f"Aquí tienes {label}. "
            "Ajusta el encuadre, el zoom y la banda en el visor como prefieras. "
            "Cuando estés listo, dime qué quieres analizar"
        )
        tail = (
            ": morfología completa, segmentación e isofotas, fotometría básica, o perfil de brillo."
        )
    return text + (tail if list_analyses else ".")


def segmentation_done_message(label: str, lang: str = "es") -> str:
    if lang == "en":
        return (
            f"Segmentation of {label} completed. "
            "The mask of the detected source is saved in the artifacts."
        )
    return (
        f"Segmentación de {label} completada. "
        "La máscara de la fuente detectada está guardada en los artefactos."
    )


def image_loaded_message(label: str, lang: str = "es") -> str:
    if lang == "en":
        return f"Image of {label} loaded successfully."
    return f"Imagen de {label} cargada correctamente."


def analysis_done_message(label: str, lang: str = "es") -> str:
    if lang == "en":
        return f"Analysis of {label} completed. See the artifacts for the results."
    return f"Análisis de {label} completado. Consulta los artefactos para ver los resultados."


def no_summary_message(lang: str = "es") -> str:
    if lang == "en":
        return "The agent finished without writing a summary."
    return "El agente completó la ejecución sin generar un resumen."


def isophotes_unavailable_message(lang: str = "es") -> str:
    if lang == "en":
        return "Isophotes are not available for this image."
    return "Isofotas no disponibles para esta imagen."


def catalog_and_field(catalog: str, size_arcmin: float, lang: str = "es") -> str:
    if lang == "en":
        return f"Catalogue used: {catalog}. Field: {size_arcmin:.1f} arcmin."
    return f"Catálogo usado: {catalog}. Campo: {size_arcmin:.1f} arcmin."
