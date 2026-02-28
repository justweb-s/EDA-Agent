"""Critic models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class CriticReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    quality_score: float
    verdict: Literal["ok", "needs_improvement"]
    missing_analyses: list[str] = Field(default_factory=list)
    shallow_interpretations: list[str] = Field(default_factory=list)
    proposed_steps: list[str] = Field(default_factory=list)
