"""Human-in-the-loop routes.

These endpoints will be implemented after interrupts/resume are wired in.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/sessions", tags=["hitl"])


@router.get("/{session_id}/interrupt")
async def get_interrupt(session_id: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.post("/{session_id}/resume")
async def resume_session(session_id: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
