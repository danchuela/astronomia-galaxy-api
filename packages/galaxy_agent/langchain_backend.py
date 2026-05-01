from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from openai import OpenAI

from packages.galaxy_agent.domain.models import Target, TaskType
from packages.galaxy_agent.models import AnalyzeRequest
from packages.galaxy_agent.request_utils import last_user_message
from packages.galaxy_core.domain.imaging import (
    get_capabilities_description,
)

logger = logging.getLogger(__name__)


def _parse_bool(value: object, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in (
            "true",
            "1",
            "yes",
            "si",
        )
    return default


def _extract_catalog_from_text(text: str) -> str | None:
    normalized = " ".join(text.upper().replace("-", " ").replace("_", " ").split())
    if not normalized:
        return None

    if "DSS2 BLUE" in normalized or "DSS2 AZUL" in normalized:
        return "DSS2-BLUE"
    if "DSS2 IR" in normalized or "DSS2 INFRARED" in normalized or "DSS2 INFRARROJO" in normalized:
        return "DSS2-IR"
    if "2MASS J" in normalized or "2MASSJ" in normalized:
        return "2MASS-J"
    if "GALEX" in normalized:
        return "GALEX"
    if re.search(r"\bSDSS\b", normalized):
        return "SDSS"
    if re.search(r"\bDSS2\b", normalized):
        return "DSS2"
    if re.search(r"\bDSS\b", normalized):
        return "DSS"
    return None


_METRIC_TRANSLATIONS: dict[str, str] = {
    "CAS": "Parámetros CAS",
    "Concentration": "Concentración (C)",
    "Asymmetry": "Asimetría (A)",
    "Smoothness": "Suavidad (S)",
    "merger score": "puntuación de fusión",
    "bulge score": "puntuación de bulbo",
    "Sérsic index": "Índice de Sérsic",
    "r_half": "r_half",
    "Radial brightness profile": "Perfil radial de brillo",
    "annuli": "anillos",
    "Peak mean flux": "Flujo medio máximo",
    "CAS metrics unavailable": "Parámetros CAS: no disponibles",
}


def _format_metrics_spanish(morphology_summary: str) -> str:
    """Translate English metric labels to Spanish deterministically."""
    result = morphology_summary
    for eng, spa in _METRIC_TRANSLATIONS.items():
        result = result.replace(eng, spa)
    return result


class LangChainBackend:
    def __init__(self) -> None:
        self._parse_model = os.getenv("OPENAI_PARSE_MODEL", "gpt-4.1")
        self._model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self._client = OpenAI()

    def _build_system_prompt(self, viewer_context: str | None = None) -> str:
        capabilities = get_capabilities_description()
        return (
            "Eres el parser de un sistema de astronomía "
            "interactivo. El sistema muestra objetos "
            "celestes en un visor interactivo (Aladin) "
            "y puede ejecutar análisis de imagen sobre "
            "lo que el usuario está viendo.\n\n"
            "CAPACIDADES:\n"
            "- Visualizar cualquier objeto celeste por "
            "nombre (M87, NGC 1300, Andrómeda...) o por "
            "coordenadas ecuatoriales (RA/Dec).\n"
            "- Elegir banda/survey: visible, infrarrojo, "
            "uv. O catálogo concreto: SDSS, DSS, DSS2, "
            "DSS2-BLUE, GALEX, 2MASS-J.\n"
            "- Analizar la imagen visible en el visor: "
            "segmentación e isofotas, fotometría básica, "
            "morfología completa y perfil de brillo.\n\n"
            "NO PUEDE: espectros, contar estrellas, "
            "responder preguntas generales de astronomía, "
            "nada ajeno a visualización y análisis.\n\n"
            "REGLAS PARA can_fulfill:\n"
            "- can_fulfill = TRUE siempre que el usuario "
            "mencione un objeto celeste (nombre o "
            "coordenadas RA/Dec), pida visualizarlo, "
            "analizarlo, o continúe una conversación "
            "sobre uno. Incluye casos como:\n"
            "  'M31', 'RA 187.7 Dec 12.4', "
            "'muéstrame andrómeda', 'analiza esto', "
            "'perfil de brillo', 'listo', 'procede', "
            "'¿dónde está M87?', '¿dónde queda NGC 891?', "
            "'localiza M51', 'encuádrame en la Vía Láctea', "
            "'ahora NGC 891', 'ahora 891', 'cambia a M101'.\n"
            "- can_fulfill = FALSE en estos casos:\n"
            "  1. SOLO preguntas de seguimiento (interrogativas) "
            "sobre un objeto ya visto o analizado: '¿qué tipo "
            "de galaxia es?', '¿cuál es su morfología?', '¿es "
            "una merger?', '¿en qué banda era?', '¿qué catálogo "
            "usaste?'. En este caso usa decline_reason para "
            "RESPONDER directamente con la información del "
            "historial. IMPORTANTE: si el usuario pide "
            "EJECUTAR un análisis con un imperativo ('hazme', "
            "'haz', 'analiza', 'calcula', 'dame el análisis', "
            "'quiero el perfil', 'repite', 'vuelve a'), "
            "can_fulfill = TRUE aunque ya se haya hecho antes. "
            "Un imperativo siempre lanza el task, nunca se "
            "responde desde el historial.\n"
            "  2. Acciones del visor interactivo (frontend): "
            "'cambia el survey a X', 'cambia a DSS', 'pon "
            "zoom +5', 'ajusta el zoom', 'acerca más'. "
            "decline_reason = 'Usa los botones del visor "
            "interactivo (encima de este chat) para cambiar "
            "el survey o el zoom.'\n"
            "  3. Peticiones claramente fuera de alcance: "
            "'dame el espectro', 'cuántas estrellas', "
            "'qué es un quásar'. decline_reason breve en "
            "español.\n"
            "- can_fulfill = FALSE si pregunta por "
            "las capacidades del sistema. Responde con:\n"
            "[CAPABILITIES]\n"
            f"{capabilities}\n"
            "Reformula de forma natural, no copies "
            "literalmente.\n\n"
            "Cuando can_fulfill es false: decline_reason "
            "en español, breve y natural.\n"
            "Cuando can_fulfill es true: "
            "decline_reason = null.\n\n"
            "DETERMINACIÓN DEL TASK — elige la tarea MÁS ESPECÍFICA que pide el usuario:\n"
            "- null: solo quiere ver el objeto, sin análisis.\n"
            "- 'segment': pide segmentación, contorno, detección del objeto.\n"
            "- 'isophotes': pide isofotas, contornos de brillo, elipses isofotales.\n"
            "- 'cas': pide SOLO concentración/asimetría/suavidad (C, A, S o CAS).\n"
            "- 'radial_profile': pide SOLO el perfil radial, la curva de brillo radial, "
            "distribución de luz por anillos o cómo cae el brillo con la distancia.\n"
            "- 'sersic': pide SOLO el índice de Sérsic, el perfil de Sérsic o r_half.\n"
            "- 'measure_basic': pide fotometría, área, elipticidad, medidas básicas "
            "(sin especificar un único parámetro concreto).\n"
            "- 'morphology_summary': pide morfología completa, análisis exhaustivo, "
            "todos los parámetros, clasificación morfológica o un análisis general. "
            "SOLO usar cuando el usuario quiera TODO, no cuando pida algo específico.\n\n"
            "NÚMEROS DE CATÁLOGO: un número de 2-5 dígitos "
            "escrito solo (como '891', '4038', '1300', '51') "
            "es un número NGC o Messier. Trátalo como el "
            "nombre de un objeto: '891' → 'NGC 891', "
            "'4038' → 'NGC 4038', '51' → 'M51'. "
            "Siempre devuélvelo como nombre estándar en "
            "el campo name.\n\n"
            "COORDENADAS: si el usuario escribe "
            "'RA X Dec Y' o 'X, Y' o cualquier par de "
            "números que parezcan coordenadas celestes, "
            "extrae ra_deg y dec_deg. Si menciona "
            "'campo', 'zoom', 'arcmin' extrae size_arcmin.\n\n"
            "INFERENCIA DEL OBJETO: si el mensaje actual "
            "no menciona objeto pero hace referencia a "
            "uno anterior ('visualízalo', 'listo', "
            "'procede', 'quiero verlo'), infiere el "
            "objeto del historial.\n\n"
            "PARÁMETROS DE ANÁLISIS (controles de precisión):\n"
            "El usuario puede pedir ajustes en lenguaje natural. Extrae:\n"
            "- thresh_sigma: número o null. Umbral SEP de detección. "
            "Frases → valor: 'más ajustado'/'solo el núcleo'/'muy estricto' → 3.0-4.0, "
            "'normal'/'por defecto' → 2.0, 'muy sensible'/'coge todo'/'halo completo' → 1.5. "
            "También extrae valores explícitos: 'umbral 2.5 sigma' → 2.5.\n"
            "- n_iso: entero o null. Número de contornos/isofotas (rango 4-20). "
            "Frases: 'más isofotas'/'más detalle' → 12, 'pocas isofotas' → 4, "
            "'10 isofotas' → 10.\n"
            "- n_bins: entero o null. Resolución del perfil radial (rango 10-50). "
            "Frases: 'más resolución radial'/'más detalle radial' → 30, "
            "'perfil grueso' → 10, '20 anillos' → 20.\n\n"
            "Devuelve SOLO un JSON con:\n"
            "- can_fulfill: boolean\n"
            "- decline_reason: string | null\n"
            "- name: string | null (nombre astronómico "
            "estándar: número Messier como M31, número "
            "NGC/IC, o nombre común en inglés como "
            "Andromeda, Whirlpool, Sombrero. NUNCA "
            "traducir ni usar tildes.)\n"
            "- ra_deg, dec_deg: number | null\n"
            "- catalog: string | null\n"
            "- band: 'visible' | 'infrared' | 'uv' | null\n"
            "- size_arcmin: number (defecto 10.0)\n"
            "- task: 'segment' | 'measure_basic' | 'morphology_summary' | "
            "'cas' | 'radial_profile' | 'sersic' | 'isophotes' | null\n"
            "- thresh_sigma: number | null\n"
            "- n_iso: integer | null\n"
            "- n_bins: integer | null\n"
            "Si hay catálogo explícito, úsalo en vez de "
            "band. Si hay nombre y coordenadas, "
            "devuelve ambos.\n"
            + (
                f"\nVISTA DEL VISOR ACTIVA: RA {viewer_context}\n"
                "Usa estas coordenadas del visor (ra_deg/dec_deg) SOLO si el último "
                "mensaje del usuario no menciona ningún nombre de objeto ni número "
                "astronómico. Si el último mensaje contiene un nombre ('M51', 'NGC 891', "
                "'Andrómeda') o un número suelto ('891', '4038'), extrae ESE objeto "
                "como name y devuelve ra_deg=null, dec_deg=null.\n"
                if viewer_context
                else ""
            )
        )

    def enrich_request(
        self,
        request: AnalyzeRequest,
    ) -> AnalyzeRequest:
        if request.target is not None and request.task is not None:
            return request

        messages = request.get_normalized_messages()
        if not messages:
            return request

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for "
                "natural language requests. "
                "Set it in .env and restart the "
                "container (or process)."
            )

        # request.message is always the current user turn.
        # request.messages (get_normalized_messages) is the prior history — it does NOT include
        # request.message. Use them separately so the LLM always sees the current message.
        current_message = (request.message or "").strip() or last_user_message(request)
        history_text = "\n".join(f"{m.role}: {m.content}" for m in messages)
        last_user_text = current_message

        viewer_ctx: str | None = None
        if request.view_ra_deg is not None and request.view_hips_id:
            viewer_ctx = (
                f"{request.view_ra_deg:.4f}°, Dec {request.view_dec_deg:.4f}°, "
                f"survey {request.view_hips_id}, campo {request.view_size_arcmin:.1f}'"
            )

        system_prompt = self._build_system_prompt(viewer_context=viewer_ctx)

        # Separate current message from history so the LLM cannot mistake
        # a name in the history (e.g. "M87") for the object in the current message.
        user_content = (
            (
                f"Historial (solo para referencias implícitas como 'analiza esto'):\n"
                f"{history_text}\n\n"
                f"Mensaje actual: {last_user_text}"
            )
            if history_text
            else last_user_text
        )

        response = self._client.chat.completions.create(
            model=self._parse_model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            timeout=30,
        )

        if not response.choices:
            return request
        content = response.choices[0].message.content or "{}"
        try:
            data: dict[str, Any] = json.loads(content)
        except json.JSONDecodeError:
            logger.warning(
                "Failed to parse LLM JSON response: %s",
                content,
            )
            return request

        can_fulfill = _parse_bool(
            data.get("can_fulfill", True),
            default=True,
        )
        decline_reason = data.get("decline_reason")
        if isinstance(decline_reason, str):
            decline_reason = decline_reason.strip() or None
        if not can_fulfill and decline_reason:
            target = request.target
            if target is None and data.get("name"):
                target = Target(name=str(data["name"]))
            return AnalyzeRequest(
                request_id=request.request_id,
                message=request.message,
                messages=request.messages,
                target=target or Target(name=""),
                task="fetch_image",
                image_url=request.image_url,
                options=request.options or {},
                out_of_scope=True,
                decline_message=decline_reason,
            )

        name = data.get("name")
        ra_deg = data.get("ra_deg")
        dec_deg = data.get("dec_deg")
        catalog = data.get("catalog")
        band = data.get("band")
        size_arcmin = 10.0
        if data.get("size_arcmin") is not None:
            try:
                size_arcmin = float(
                    data["size_arcmin"],
                )
            except (TypeError, ValueError):
                logger.debug(
                    "enrich_request: invalid size_arcmin=%r, using default", data.get("size_arcmin")
                )

        options = dict(request.options) if request.options else {}
        catalog_from_text = _extract_catalog_from_text(
            last_user_text,
        )

        if catalog is not None and str(catalog).strip():
            options["catalog"] = str(catalog).strip()
        if catalog_from_text:
            options["catalog"] = catalog_from_text

        if ra_deg is not None and dec_deg is not None:
            try:
                options["ra_deg"] = float(ra_deg)
                options["dec_deg"] = float(dec_deg)
            except (TypeError, ValueError):
                logger.debug(
                    "enrich_request: invalid coordinates ra=%r dec=%r, omitting", ra_deg, dec_deg
                )
        if band:
            options["band"] = str(band)
        if options.get("catalog"):
            options.pop("band", None)
        options.setdefault("size_arcmin", size_arcmin)

        # Analysis control parameters from LLM output
        for key, cast in (("thresh_sigma", float), ("n_iso", int), ("n_bins", int)):
            raw = data.get(key)
            if raw is not None:
                try:
                    options[key] = cast(raw)
                except (TypeError, ValueError):
                    logger.debug("enrich_request: invalid value for %s=%r, omitting", key, raw)

        target = request.target
        if target is None and name:
            target = Target(name=str(name))

        valid_tasks = {
            "segment",
            "measure_basic",
            "morphology_summary",
            "cas",
            "radial_profile",
            "sersic",
            "isophotes",
        }
        task_str = str(data.get("task") or "")
        task: TaskType = task_str if task_str in valid_tasks else "resolve"  # type: ignore[assignment]

        return AnalyzeRequest(
            request_id=request.request_id,
            message=request.message,
            messages=request.messages,
            target=target,
            task=task,
            image_url=request.image_url,
            options=options,
            view_ra_deg=request.view_ra_deg,
            view_dec_deg=request.view_dec_deg,
            view_size_arcmin=request.view_size_arcmin,
            view_hips_id=request.view_hips_id,
            image_data=request.image_data,
        )

    def generate_accompanying_summary(
        self,
        target_name: str,
        band: str | None,
        morphology_summary: str,
        user_message: str | None = None,
        simbad_morph_type: str | None = None,
    ) -> str:
        """Build analysis summary: LLM writes qualitative intro, metrics appended verbatim."""
        metrics_block = _format_metrics_spanish(morphology_summary)

        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            return metrics_block

        band_info = f" en banda {band}" if band else ""
        user_text = (user_message or "").strip() or f"análisis de {target_name}{band_info}"
        morph_line = (
            f"Tipo morfológico de la literatura (SIMBAD): {simbad_morph_type}\n\n"
            if simbad_morph_type
            else ""
        )
        prompt = (
            f"El usuario dijo: \xab{user_text}\xbb.\n\n"
            f"{morph_line}"
            "Resultado del análisis morfológico:\n"
            f"{morphology_summary}\n\n"
            "Escribe UNA O DOS frases introductorias en español que resuman "
            "los resultados morfológicos y la fiabilidad del análisis. "
            "Si se proporciona el tipo morfológico de SIMBAD, menciónalo como referencia. "
            "NO incluyas ningún valor numérico, parámetro CAS, "
            "Sérsic, ni ninguna cifra. Solo una introducción cualitativa breve."
        )
        try:
            resp = self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": "Responde solo con la frase introductoria. Nada más.",
                    },
                    {"role": "user", "content": prompt},
                ],
                timeout=30,
            )
            if not resp.choices:
                return metrics_block
            intro = (resp.choices[0].message.content or "").strip()
            if not intro:
                return metrics_block
            if band and "banda" not in intro.lower():
                intro = f"Análisis de {target_name} en banda {band}. {intro}"
            return f"{intro}\n\n{metrics_block}"
        except Exception:
            logger.warning("LLM summary generation failed", exc_info=True)
            return metrics_block
