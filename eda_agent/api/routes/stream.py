"""SSE stream route.

This will be implemented after the LangGraph graph is available.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import StreamingResponse

router = APIRouter(prefix="/sessions", tags=["stream"])


@router.get("/{session_id}/stream")
async def stream_session(
    session_id: str, request: Request, from_event: int = 0
) -> StreamingResponse:
    store = getattr(request.app.state, "session_store", None)
    if store is None or store.get(session_id) is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    broker = request.app.state.sse_broker
    channel = await broker.get_channel(session_id)

    async def event_gen() -> AsyncIterator[bytes]:
        async for msg in channel.subscribe(from_event=from_event):
            payload = json.dumps(msg.data, ensure_ascii=False)
            chunk = f"id: {msg.message_id}\nevent: {msg.event}\ndata: {payload}\n\n"
            yield chunk.encode("utf-8")
            if msg.event in {"session_completed", "session_failed", "session_suspended"}:
                return

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
