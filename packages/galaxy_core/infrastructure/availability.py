"""Remember external services that just failed, so requests skip them for a while.

After the CDS incident (Sesame and SIMBAD offline since 2026-10-28) every lookup waited for
the CDS timeout before trying a mirror. A service that fails is skipped for COOL_DOWN
seconds (env SERVICE_COOL_DOWN_SECONDS, default 300); the next request after that tries it
again. State is per process and in memory: a restart simply tries every service again.
"""

from __future__ import annotations

import os
import threading
import time

_lock = threading.Lock()
_down_until: dict[str, float] = {}


def _cool_down() -> float:
    try:
        value = float(os.environ.get("SERVICE_COOL_DOWN_SECONDS", "300"))
    except ValueError:
        return 300.0
    return value if value >= 0 else 300.0


def is_available(service: str) -> bool:
    with _lock:
        until = _down_until.get(service)
        if until is None:
            return True
        if time.monotonic() >= until:
            del _down_until[service]
            return True
        return False


def mark_down(service: str) -> None:
    with _lock:
        _down_until[service] = time.monotonic() + _cool_down()


def mark_up(service: str) -> None:
    with _lock:
        _down_until.pop(service, None)


def reset() -> None:
    """Forget every failure (tests)."""
    with _lock:
        _down_until.clear()
