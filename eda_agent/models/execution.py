"""Execution models."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class ExecutionResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    stdout: str = ""
    stderr: str = ""
    success: bool
    outputs: list[dict] = Field(default_factory=list)


class ExecutionSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    section: str
    findings: str
    key_statistics: dict[str, float | int | str] = Field(default_factory=dict)
    charts_produced: list[str] = Field(default_factory=list)
    anomalies_found: list[str] = Field(default_factory=list)
    columns_analyzed: list[str] = Field(default_factory=list)
    created_at: datetime
