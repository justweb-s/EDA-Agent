"""Single-user session store.

This is a lightweight persistence mechanism to support the API layer while the
LangGraph pipeline is being implemented.

Records are stored as JSON files under OUTPUT_DIR/sessions.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from eda_agent.models.dataset import DatasetContext
from eda_agent.models.notebook import NotebookCell


@dataclass(frozen=True)
class SessionRecord:
    session_id: str
    status: str
    created_at: datetime
    file_name: str
    file_path: str
    dataset_context: DatasetContext
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o"
    mode: str = "auto"
    hitl_enabled: bool = False
    user_instructions: str | None = None
    n_cells: int = 0
    notebook_cells: list[NotebookCell] | None = None
    notebook_path: str | None = None
    error: str | None = None


class SessionStore:
    def __init__(self, *, output_dir: Path) -> None:
        self._base_dir = (output_dir / "sessions").resolve()
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def _path_for(self, session_id: str) -> Path:
        return self._base_dir / f"{session_id}.json"

    def upsert(self, record: SessionRecord) -> None:
        payload = {
            "session_id": record.session_id,
            "status": record.status,
            "created_at": record.created_at.isoformat(),
            "file_name": record.file_name,
            "file_path": record.file_path,
            "dataset_context": record.dataset_context.model_dump(mode="json"),
            "llm_provider": record.llm_provider,
            "llm_model": record.llm_model,
            "mode": record.mode,
            "hitl_enabled": record.hitl_enabled,
            "user_instructions": record.user_instructions,
            "n_cells": record.n_cells,
            "notebook_cells": [c.model_dump(mode="json") for c in (record.notebook_cells or [])],
            "notebook_path": record.notebook_path,
            "error": record.error,
        }

        path = self._path_for(record.session_id)
        tmp = path.with_suffix(".json.tmp")

        with self._lock:
            tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            tmp.replace(path)

    def get(self, session_id: str) -> SessionRecord | None:
        path = self._path_for(session_id)
        if not path.exists():
            return None

        with self._lock:
            payload = json.loads(path.read_text(encoding="utf-8"))

        notebook_cells = [NotebookCell.model_validate(c) for c in payload.get("notebook_cells", [])]

        return SessionRecord(
            session_id=str(payload["session_id"]),
            status=str(payload["status"]),
            created_at=datetime.fromisoformat(payload["created_at"]),
            file_name=str(payload["file_name"]),
            file_path=str(payload["file_path"]),
            dataset_context=DatasetContext.model_validate(payload["dataset_context"]),
            llm_provider=str(payload.get("llm_provider") or "openai"),
            llm_model=str(payload.get("llm_model") or "gpt-4o"),
            mode=str(payload.get("mode") or "auto"),
            hitl_enabled=bool(payload.get("hitl_enabled") or False),
            user_instructions=payload.get("user_instructions"),
            n_cells=int(payload.get("n_cells", 0)),
            notebook_cells=notebook_cells,
            notebook_path=payload.get("notebook_path"),
            error=payload.get("error"),
        )

    def list(self) -> list[SessionRecord]:
        records: list[SessionRecord] = []

        with self._lock:
            paths = sorted(self._base_dir.glob("*.json"))

        for path in paths:
            payload = json.loads(path.read_text(encoding="utf-8"))

            notebook_cells = [
                NotebookCell.model_validate(c) for c in payload.get("notebook_cells", [])
            ]
            records.append(
                SessionRecord(
                    session_id=str(payload["session_id"]),
                    status=str(payload["status"]),
                    created_at=datetime.fromisoformat(payload["created_at"]),
                    file_name=str(payload["file_name"]),
                    file_path=str(payload["file_path"]),
                    dataset_context=DatasetContext.model_validate(payload["dataset_context"]),
                    llm_provider=str(payload.get("llm_provider") or "openai"),
                    llm_model=str(payload.get("llm_model") or "gpt-4o"),
                    mode=str(payload.get("mode") or "auto"),
                    hitl_enabled=bool(payload.get("hitl_enabled") or False),
                    user_instructions=payload.get("user_instructions"),
                    n_cells=int(payload.get("n_cells", 0)),
                    notebook_cells=notebook_cells,
                    notebook_path=payload.get("notebook_path"),
                    error=payload.get("error"),
                )
            )

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records

    def delete(self, session_id: str) -> bool:
        path = self._path_for(session_id)
        if not path.exists():
            return False

        with self._lock:
            path.unlink(missing_ok=True)

        return True


def new_session_record(
    *,
    session_id: str,
    file_name: str,
    file_path: str,
    dataset_context: DatasetContext,
    llm_provider: str = "openai",
    llm_model: str = "gpt-4o",
    mode: str = "auto",
    hitl_enabled: bool = False,
    user_instructions: str | None = None,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        status="created",
        created_at=datetime.now(UTC),
        file_name=file_name,
        file_path=file_path,
        dataset_context=dataset_context,
        llm_provider=llm_provider,
        llm_model=llm_model,
        mode=mode,
        hitl_enabled=hitl_enabled,
        user_instructions=user_instructions,
        n_cells=0,
    )
