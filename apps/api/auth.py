from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status

from apps.api.config import Settings, get_settings


def verify_api_key(
    settings: Annotated[Settings, Depends(get_settings)],
    x_api_key: Annotated[str | None, Header(alias="X-API-Key")] = None,
) -> None:
    if not settings.require_api_key:
        return

    if not settings.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "API_KEY_NOT_CONFIGURED",
                "message": "API key is required but missing.",
            },
        )

    if x_api_key is None or not secrets.compare_digest(x_api_key, settings.api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail={"code": "INVALID_API_KEY", "message": "Invalid X-API-Key header."},
        )
