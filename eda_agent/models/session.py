"""Session models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict


class SessionMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    session_id: str
    started_at: datetime
    llm_provider: str
    llm_model: str
    file_name: str


SessionStatus = Literal["created", "in_progress", "suspended", "completed", "failed", "abandoned"]
