"""Parent graph state definition."""

from __future__ import annotations

from typing import Literal, TypedDict

from langchain_core.messages import AnyMessage

from eda_agent.models.critic import CriticReview
from eda_agent.models.dataset import DatasetContext
from eda_agent.models.execution import ExecutionSummary
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata


class EDAState(TypedDict, total=False):
    messages: list[AnyMessage]
    dataset_context: DatasetContext
    eda_plan: list[EDAStep]
    current_step_index: int
    notebook_cells: list[NotebookCell]
    execution_history: list[ExecutionSummary]
    retry_count: int
    critic_feedback: CriticReview | None
    mode: Literal["auto", "chat"]
    hitl_enabled: bool
    session_metadata: SessionMetadata
    final_notebook_path: str | None
