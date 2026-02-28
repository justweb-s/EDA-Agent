"""SSE stream route.

This will be implemented after the LangGraph graph is available.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/sessions", tags=["stream"])


@router.get("/{session_id}/stream")
async def stream_session(session_id: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
