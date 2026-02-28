"""FastAPI request/response schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DatasetContextSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    file_name: str
    n_rows: int
    n_columns: int
    detected_issues: list[dict] = Field(default_factory=list)


class SessionCreateResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: Literal["created"]
    dataset_context_summary: DatasetContextSummary


class SessionRecordResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    status: str
    created_at: datetime
    file_name: str
    file_path: str
    n_cells: int = 0


class SessionListResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    sessions: list[SessionRecordResponse]
