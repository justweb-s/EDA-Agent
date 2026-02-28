"""Health check routes."""

from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str]:
    config = getattr(request.app.state, "config", None)
    env = getattr(config, "environment", "unknown")
    return {"status": "ok", "environment": str(env)}


@router.get("/health/kernel")
async def kernel_health() -> dict[str, str]:
    # The kernel subsystem will be implemented in the next milestones.
    return {"status": "ok"}
