"""EDA Loop subgraph state."""

from __future__ import annotations

from typing import TypedDict

from eda_agent.models.dataset import DatasetContext
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.plan import EDAStep


class EDALoopState(TypedDict, total=False):
    current_step: EDAStep
    dataset_context: DatasetContext
    generated_code: str
    execution_result: ExecutionResult
    observer_verdict: str
    retry_messages: list[str]
    local_retry_count: int
