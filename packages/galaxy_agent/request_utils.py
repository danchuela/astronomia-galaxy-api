from __future__ import annotations

from packages.galaxy_agent.models import AnalyzeRequest


def last_user_message(request: AnalyzeRequest) -> str:
    messages = request.get_normalized_messages()
    if messages:
        last_user = next((m.content for m in reversed(messages) if m.role == "user"), None)
        if last_user:
            return last_user
    return request.message or ""


def append_catalog_and_field(summary: str, catalog: str, size_arcmin: float) -> str:
    base = summary.rstrip().rstrip(".")
    return f"{base}. Catálogo usado: {catalog}. Campo: {size_arcmin:.1f} arcmin."
