from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache


def _to_bool(value: str, default: bool) -> bool:
    if not value:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class Settings:
    api_key: str
    require_api_key: bool
    artifact_dir: str
    log_level: str
    langsmith_api_key: str
    langsmith_tracing: bool


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        api_key=os.getenv("API_KEY", ""),
        require_api_key=_to_bool(os.getenv("REQUIRE_API_KEY", "true"), True),
        artifact_dir=os.getenv("ARTIFACT_DIR", "artifacts"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        langsmith_api_key=os.getenv("LANGSMITH_API_KEY", ""),
        langsmith_tracing=_to_bool(os.getenv("LANGSMITH_TRACING", "false"), False),
    )
