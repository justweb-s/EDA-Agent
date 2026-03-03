"""Human-in-the-loop routes.

These endpoints will be implemented after interrupts/resume are wired in.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, ConfigDict

from eda_agent.checkpointing.setup import build_checkpointer
from eda_agent.config import EDAConfig
from eda_agent.graph.parent.supervisor import build_supervisor_graph

from ..session_runner import run_minimal_session
from ..session_store import SessionStore

router = APIRouter(prefix="/sessions", tags=["hitl"])


class ResumeRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    resume: dict[str, Any]


@router.get("/{session_id}/interrupt")
async def get_interrupt(session_id: str, request: Request) -> dict[str, Any]:
    store = cast(SessionStore | None, getattr(request.app.state, "session_store", None))
    if store is None or store.get(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    config = cast(EDAConfig, request.app.state.config)
    checkpointer = build_checkpointer(config)
    supervisor = build_supervisor_graph(checkpointer=checkpointer)
    runnable_config = {"configurable": {"thread_id": session_id}}

    try:
        snapshot = await asyncio.to_thread(supervisor.get_state, runnable_config)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="No interrupt state available",
        ) from None
    finally:
        try:
            conn = getattr(checkpointer, "conn", None)
            if conn is not None:
                await asyncio.to_thread(conn.close)
        except Exception:
            pass

    interrupts: list[dict[str, Any]] = []
    for task in getattr(snapshot, "tasks", ()) or ():
        for it in getattr(task, "interrupts", ()) or ():
            interrupts.append(
                {
                    "value": getattr(it, "value", it),
                    "resumable": getattr(it, "resumable", None),
                    "ns": getattr(it, "ns", None),
                    "when": getattr(it, "when", None),
                }
            )

    if not interrupts:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No interrupt pending")

    return {
        "session_id": session_id,
        "interrupts": interrupts,
        "next": list(getattr(snapshot, "next", ()) or ()),
    }


@router.post("/{session_id}/resume")
async def resume_session(
    session_id: str,
    request: Request,
    payload: ResumeRequest,
) -> dict[str, Any]:
    store = cast(SessionStore | None, getattr(request.app.state, "session_store", None))
    if store is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    rec = store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    if rec.status != "suspended":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Session is not suspended (status={rec.status})",
        )

    store.upsert(replace(rec, status="running", error=None))
    asyncio.create_task(
        run_minimal_session(app=request.app, session_id=session_id, resume=payload.resume)
    )

    return {"session_id": session_id, "status": "resuming"}
