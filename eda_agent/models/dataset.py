"""Dataset-related domain models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ColumnInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str
    n_null: int
    n_unique: int
    sample_values: list[str] = Field(default_factory=list)
    detected_semantic_type: str = "unknown"


class BasicStats(BaseModel):
    model_config = ConfigDict(frozen=True)

    numeric: dict[str, dict[str, float | int | None]] = Field(default_factory=dict)
    categorical: dict[str, dict[str, int]] = Field(default_factory=dict)


class DetectedIssueModel(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    severity: Literal["warning", "error"]
    column: str | None = None


class DatasetContext(BaseModel):
    """Immutable dataset context shared across all agents."""

    model_config = ConfigDict(frozen=True)

    file_path: str
    file_name: str
    shape: tuple[int, int]
    columns: list[ColumnInfo]
    memory_usage_mb: float
    preview_markdown: str
    basic_stats: BasicStats
    detected_issues: list[DetectedIssueModel] = Field(default_factory=list)
    created_at: datetime

    def to_prompt_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dict intended for prompt injection."""

        return self.model_dump(mode="json")
