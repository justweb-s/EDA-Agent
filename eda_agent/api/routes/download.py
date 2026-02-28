"""Notebook download routes.

Will be implemented once notebook assembly is wired in.
"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import Response

from eda_agent.api.notebook_export import export_ipynb_bytes
from eda_agent.config import EDAConfig

router = APIRouter(prefix="/sessions", tags=["download"])


@router.get("/{session_id}/notebook")
async def download_notebook(session_id: str, request: Request) -> Response:
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    rec = store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if rec.status != "completed":
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Notebook not available")

    config = cast(EDAConfig, request.app.state.config)
    default_path = (config.output_dir / "notebooks" / f"eda-agent-{session_id}.ipynb").resolve()
    notebook_path = Path(rec.notebook_path).resolve() if rec.notebook_path else default_path

    if notebook_path.exists():
        data = notebook_path.read_bytes()
    else:
        cells = rec.notebook_cells or []
        data = export_ipynb_bytes(cells)

    filename = f"eda-agent-{session_id}.ipynb"
    return Response(
        content=data,
        media_type="application/x-ipynb+json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/{session_id}/notebook/cells")
async def get_notebook_cells(session_id: str, request: Request) -> list[dict]:
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    rec = store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return [c.model_dump(mode="json") for c in (rec.notebook_cells or [])]
