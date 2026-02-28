"""Critic subgraph state."""

from __future__ import annotations

from typing import TypedDict

from eda_agent.models.critic import CriticReview
from eda_agent.models.dataset import DatasetContext
from eda_agent.models.notebook import NotebookCell


class CriticState(TypedDict, total=False):
    notebook_cells: list[NotebookCell]
    dataset_context: DatasetContext
    section_name: str
    critic_feedback: CriticReview
