"""EDA Loop subgraph state."""

from __future__ import annotations

from typing import TypedDict

from eda_agent.models.dataset import DatasetContext
from eda_agent.models.execution import ExecutionResult, ExecutionSummary
from eda_agent.models.notebook import CellOutput, NotebookCell
from eda_agent.models.plan import EDAStep


class EDALoopState(TypedDict, total=False):
    eda_plan: list[EDAStep]
    current_step_index: int
    current_step: EDAStep
    dataset_context: DatasetContext
    execution_history: list[ExecutionSummary]
    generated_code: str
    execution_count: int | None
    cell_outputs: list[CellOutput]
    execution_result: ExecutionResult
    observer_verdict: str
    retry_messages: list[str]
    local_retry_count: int
    notebook_cells: list[NotebookCell]
    execution_summary: ExecutionSummary
