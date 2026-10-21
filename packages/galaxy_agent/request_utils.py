from __future__ import annotations

from packages.galaxy_agent.messages import catalog_and_field
from packages.galaxy_agent.models import AnalyzeRequest


def last_user_message(request: AnalyzeRequest) -> str:
    messages = request.get_normalized_messages()
    if messages:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if last_user:
            return last_user
    return request.message or ""


def append_catalog_and_field(
    summary: str, catalog: str, size_arcmin: float, lang: str = "es"
) -> str:
    base = summary.rstrip().rstrip(".")
    return f"{base}. {catalog_and_field(catalog, size_arcmin, lang)}"
