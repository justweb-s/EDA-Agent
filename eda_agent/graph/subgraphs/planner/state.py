"""Planner subgraph state."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AnyMessage

from eda_agent.models.dataset import DatasetContext
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata


class PlannerState(TypedDict, total=False):
    messages: list[AnyMessage]
    dataset_context: DatasetContext
    session_metadata: SessionMetadata
    dataset_analysis: str
    eda_plan: list[EDAStep]
    draft_plan: list[EDAStep]
    validation_errors: list[str]
    iteration_count: int
