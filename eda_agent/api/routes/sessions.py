"""Session management routes."""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path
from typing import Annotated, cast
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status

from eda_agent.config import EDAConfig
from eda_agent.ingestion.loader import load_file
from eda_agent.ingestion.profiler import build_dataset_context

from ..schemas import (
    DatasetContextSummary,
    SessionCreateResponse,
    SessionListResponse,
    SessionRecordResponse,
)
from ..session_runner import run_minimal_session
from ..session_store import SessionStore, new_session_record

router = APIRouter(prefix="/sessions", tags=["sessions"])


def _get_store(request: Request) -> SessionStore:
    config: EDAConfig = request.app.state.config
    store = getattr(request.app.state, "session_store", None)
    if store is None:
        store = SessionStore(output_dir=config.output_dir)
        request.app.state.session_store = store
    return cast(SessionStore, store)


@router.post("", response_model=SessionCreateResponse)
async def create_session(
    request: Request,
    file: Annotated[UploadFile, File(...)],
    provider: Annotated[str | None, Form()] = None,
    model: Annotated[str | None, Form()] = None,
    mode: Annotated[str | None, Form()] = None,
    hitl_enabled: Annotated[bool | None, Form()] = None,
    user_instructions: Annotated[str | None, Form()] = None,
) -> SessionCreateResponse:
    config: EDAConfig = request.app.state.config

    if not file.filename:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xls", ".xlsx"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type")

    llm_provider = (provider or config.llm_provider).strip().lower()
    llm_model = (model or config.llm_model).strip()
    mode_value = (mode or "auto").strip().lower()
    hitl_enabled_value = (
        bool(hitl_enabled) if hitl_enabled is not None else (config.hitl_default_mode != "none")
    )

    if llm_provider not in {"openai", "anthropic", "groq", "ollama", "mock"}:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported LLM provider"
        )
    if mode_value not in {"auto", "chat"}:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported mode")

    session_id = str(uuid4())
    stored_name = f"{session_id}{suffix}"
    stored_path = (config.upload_dir / stored_name).resolve()

    try:
        with stored_path.open("wb") as f:
            await file.seek(0)
            shutil.copyfileobj(file.file, f)
    finally:
        await file.close()

    df = load_file(stored_path)
    dataset_context = build_dataset_context(df, file_path=stored_path, config=config)

    store = _get_store(request)
    record = new_session_record(
        session_id=session_id,
        file_name=dataset_context.file_name,
        file_path=dataset_context.file_path,
        dataset_context=dataset_context,
        llm_provider=llm_provider,
        llm_model=llm_model,
        mode=mode_value,
        hitl_enabled=hitl_enabled_value,
        user_instructions=user_instructions.strip() if user_instructions else None,
    )
    store.upsert(record)

    asyncio.create_task(run_minimal_session(app=request.app, session_id=session_id))

    summary = DatasetContextSummary(
        file_name=dataset_context.file_name,
        n_rows=dataset_context.shape[0],
        n_columns=dataset_context.shape[1],
        detected_issues=[i.model_dump(mode="json") for i in dataset_context.detected_issues],
    )

    return SessionCreateResponse(
        session_id=session_id,
        status="created",
        dataset_context_summary=summary,
    )


@router.get("", response_model=SessionListResponse)
async def list_sessions(request: Request, status: str | None = None) -> SessionListResponse:
    store = _get_store(request)
    sessions = []
    for rec in store.list():
        if status and rec.status != status:
            continue
        sessions.append(
            SessionRecordResponse(
                session_id=rec.session_id,
                status=rec.status,
                created_at=rec.created_at,
                file_name=rec.file_name,
                file_path=rec.file_path,
                n_cells=rec.n_cells,
            )
        )
    return SessionListResponse(sessions=sessions)


@router.get("/{session_id}", response_model=SessionRecordResponse)
async def get_session(session_id: str, request: Request) -> SessionRecordResponse:
    store = _get_store(request)
    rec = store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    return SessionRecordResponse(
        session_id=rec.session_id,
        status=rec.status,
        created_at=rec.created_at,
        file_name=rec.file_name,
        file_path=rec.file_path,
        n_cells=rec.n_cells,
    )


@router.delete("/{session_id}")
async def delete_session(session_id: str, request: Request) -> dict[str, str]:
    config: EDAConfig = request.app.state.config
    store = _get_store(request)
    rec = store.get(session_id)
    if rec is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    deleted = store.delete(session_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    try:
        Path(rec.file_path).unlink(missing_ok=True)
    except Exception:
        pass

    try:
        default_notebook_path = (
            config.output_dir / "notebooks" / f"eda-agent-{session_id}.ipynb"
        ).resolve()
        notebook_path = (
            Path(rec.notebook_path).resolve() if rec.notebook_path else default_notebook_path
        )
        notebook_path.unlink(missing_ok=True)
    except Exception:
        pass

    return {"status": "deleted"}
