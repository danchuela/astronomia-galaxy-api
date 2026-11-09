from __future__ import annotations

import pytest

from packages.galaxy_core.infrastructure import availability


@pytest.fixture(autouse=True)
def _fresh_service_availability():
    """Each test starts with every external service available (availability.py is per process)."""
    availability.reset()
    yield
    availability.reset()
