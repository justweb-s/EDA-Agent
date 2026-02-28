"""Notebook models."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CellOutput(BaseModel):
    model_config = ConfigDict(frozen=True)

    output_type: str
    data: dict[str, str] = Field(default_factory=dict)
    text: str | None = None


class NotebookCell(BaseModel):
    model_config = ConfigDict(frozen=True)

    cell_type: Literal["code", "markdown"]
    source: str
    outputs: list[CellOutput] = Field(default_factory=list)
    execution_count: int | None = None
    step_id: str | None = None
    generated_at: datetime
    re_executable: bool = True
