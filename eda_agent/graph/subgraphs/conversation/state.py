"""Conversation subgraph state."""

from __future__ import annotations

from typing import TypedDict

from langchain_core.messages import AnyMessage

from eda_agent.models.dataset import DatasetContext
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep


class ConversationState(TypedDict, total=False):
    messages: list[AnyMessage]
    dataset_context: DatasetContext
    eda_plan: list[EDAStep]
    notebook_cells: list[NotebookCell]
    user_intent: str
    ad_hoc_code: str
    ad_hoc_result: str
