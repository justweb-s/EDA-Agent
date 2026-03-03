from __future__ import annotations

from collections.abc import Iterable

from eda_agent.models.critic import CriticReview
from eda_agent.models.dataset import DatasetContext
from eda_agent.models.execution import ExecutionResult, ExecutionSummary
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata


def coerce_dataset_context(value: object) -> DatasetContext | None:
    if value is None:
        return None
    if isinstance(value, DatasetContext):
        return value
    if isinstance(value, dict):
        return DatasetContext.model_validate(value)
    return None


def coerce_session_metadata(value: object) -> SessionMetadata | None:
    if value is None:
        return None
    if isinstance(value, SessionMetadata):
        return value
    if isinstance(value, dict):
        return SessionMetadata.model_validate(value)
    return None


def coerce_step(value: object) -> EDAStep | None:
    if value is None:
        return None
    if isinstance(value, EDAStep):
        return value
    if isinstance(value, dict):
        return EDAStep.model_validate(value)
    return None


def coerce_plan(value: object) -> list[EDAStep]:
    if value is None:
        return []
    if isinstance(value, list):
        return [(s if isinstance(s, EDAStep) else EDAStep.model_validate(s)) for s in value]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return coerce_plan(list(value))
    return []


def coerce_cells(value: object) -> list[NotebookCell]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            (c if isinstance(c, NotebookCell) else NotebookCell.model_validate(c)) for c in value
        ]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return coerce_cells(list(value))
    return []


def coerce_execution_history(value: object) -> list[ExecutionSummary]:
    if value is None:
        return []
    if isinstance(value, list):
        return [
            (s if isinstance(s, ExecutionSummary) else ExecutionSummary.model_validate(s))
            for s in value
        ]
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return coerce_execution_history(list(value))
    return []


def coerce_execution_result(value: object) -> ExecutionResult | None:
    if value is None:
        return None
    if isinstance(value, ExecutionResult):
        return value
    if isinstance(value, dict):
        return ExecutionResult.model_validate(value)
    return None


def coerce_critic_review(value: object) -> CriticReview | None:
    if value is None:
        return None
    if isinstance(value, CriticReview):
        return value
    if isinstance(value, dict):
        return CriticReview.model_validate(value)
    return None
