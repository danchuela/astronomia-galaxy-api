from __future__ import annotations

import importlib.metadata
from functools import lru_cache

from packages.galaxy_agent.constants import GALAXY_AGENT_VERSION, GALAXY_CORE_VERSION
from packages.galaxy_agent.models import Provenance


@lru_cache(maxsize=1)
def _library_versions() -> dict[str, str]:
    versions = {}
    for package in ("numpy", "sep", "statmorph", "photutils", "astropy", "scipy", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unavailable"
    return versions


def provenance_versions(langsmith_enabled: bool) -> dict[str, str]:
    return {
        **_library_versions(),
        "galaxy_core": GALAXY_CORE_VERSION,
        "galaxy_agent": GALAXY_AGENT_VERSION,
        "langsmith_enabled": str(langsmith_enabled).lower(),
    }


def build_provenance(langsmith_enabled: bool) -> Provenance:
    return Provenance(versions=provenance_versions(langsmith_enabled))


def build_stream_provenance_payload(langsmith_enabled: bool) -> dict[str, dict[str, str]]:
    return {"versions": provenance_versions(langsmith_enabled)}
