"""Planning models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

AnalysisType = Literal[
    "univariate",
    "bivariate",
    "multivariate",
    "data_quality",
    "feature_specific",
]


class EDAStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step_id: str
    section: str
    title: str
    description: str
    analysis_type: AnalysisType
    target_columns: list[str] = Field(default_factory=list)
    depends_on: list[str] = Field(default_factory=list)
    is_mandatory: bool = True
    priority: int = 0


class EDAplan(BaseModel):
    model_config = ConfigDict(frozen=True)

    steps: list[EDAStep]
