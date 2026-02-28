"""FastAPI dependencies."""

from __future__ import annotations

from fastapi import Header, HTTPException, Request, status

from eda_agent.config import EDAConfig


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """Enforce API key authentication in production.

    In development/testing this dependency is a no-op.
    """

    config: EDAConfig | None = getattr(request.app.state, "config", None)
    if config is None:
        return

    if config.environment != "production":
        return

    if not config.api_key:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="API_KEY is not configured",
        )

    if x_api_key != config.api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
