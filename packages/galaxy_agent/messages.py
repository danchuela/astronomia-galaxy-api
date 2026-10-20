from __future__ import annotations

TARGET_REQUIRED_MESSAGE = (
    "Indica de qué galaxia quieres la imagen o el análisis "
    "(por nombre, p. ej. M104, o por coordenadas). "
    "Puedes añadir la banda: visible, infrarrojo o ultravioleta."
)

ERROR_ANALYSIS_FAILED_MESSAGE = "No se pudo completar el análisis. Inténtalo de nuevo."
ERROR_IMAGE_FETCH_FAILED_MESSAGE = (
    "No se pudo obtener la imagen para esa banda en este momento. "
    "Prueba otra banda o inténtalo de nuevo."
)


def target_not_found_message(name: str) -> str:
    return (
        f'No he encontrado un objeto con el nombre "{name}". '
        "Comprueba la designación o indícame otro nombre o sus coordenadas (RA/Dec)."
    )
