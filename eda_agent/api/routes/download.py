"""Notebook download routes.

Will be implemented once notebook assembly is wired in.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

router = APIRouter(prefix="/sessions", tags=["download"])


@router.get("/{session_id}/notebook")
async def download_notebook(session_id: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")


@router.get("/{session_id}/notebook/cells")
async def get_notebook_cells(session_id: str) -> None:
    raise HTTPException(status_code=status.HTTP_501_NOT_IMPLEMENTED, detail="Not implemented")
