"""Translate LangGraph stream events into SSE events.

The full translator will be implemented once the LangGraph graph is available.
"""

from __future__ import annotations

from collections.abc import Iterable

from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep


class EventTranslator:
    def __init__(self) -> None:
        self._last_n_cells = 0

    def translate(self, event: object) -> list[tuple[str, dict]]:
        if isinstance(event, tuple) and len(event) == 2:
            first, second = event
            if first == "updates":
                event = second
            elif isinstance(second, dict):
                event = second

        if not isinstance(event, dict):
            return []

        out: list[tuple[str, dict]] = []

        if "__interrupt__" in event:
            interrupt_payload = event.get("__interrupt__")
            items: list[object]
            if isinstance(interrupt_payload, tuple):
                items = list(interrupt_payload)
            elif interrupt_payload is None:
                items = []
            else:
                items = [interrupt_payload]

            for item in items:
                value = getattr(item, "value", item)
                interrupt_type = None
                if isinstance(value, dict):
                    interrupt_type = value.get("type")

                available_actions: list[str] = []
                if interrupt_type == "plan_approval":
                    available_actions = ["approve", "reject", "modify"]

                out.append(
                    (
                        "hitl_interrupt",
                        {
                            "interrupt": {
                                "value": value,
                                "resumable": getattr(item, "resumable", None),
                                "ns": getattr(item, "ns", None),
                                "when": getattr(item, "when", None),
                            },
                            "interrupt_type": interrupt_type,
                            "data": value,
                            "available_actions": available_actions,
                        },
                    )
                )
            return out

        for _, update in event.items():
            if not isinstance(update, dict):
                continue

            if "eda_plan" in update:
                steps = _coerce_steps(update.get("eda_plan"))
                out.append(
                    (
                        "plan_generated",
                        {
                            "eda_plan": [s.model_dump(mode="json") for s in steps],
                            "n_steps": len(steps),
                        },
                    )
                )

            if "notebook_cells" in update:
                cells = _coerce_cells(update.get("notebook_cells"))
                new_cells = cells[self._last_n_cells :]
                self._last_n_cells = max(self._last_n_cells, len(cells))
                for cell in new_cells:
                    cell_payload = cell.model_dump(mode="json")
                    out.append(
                        (
                            "cell_added",
                            {
                                "cell": cell_payload,
                                "cell_type": cell_payload.get("cell_type"),
                                "content": cell_payload.get("source"),
                                "outputs": cell_payload.get("outputs", []),
                                "step_id": cell_payload.get("step_id"),
                                "execution_count": cell_payload.get("execution_count"),
                                "generated_at": cell_payload.get("generated_at"),
                                "re_executable": cell_payload.get("re_executable"),
                                "n_cells": self._last_n_cells,
                            },
                        )
                    )

            if "__interrupt__" in update:
                interrupt_payload = update.get("__interrupt__")
                if interrupt_payload is not None:
                    interrupt_type = None
                    if isinstance(interrupt_payload, dict):
                        interrupt_type = interrupt_payload.get("type")

                    available_actions = []
                    if interrupt_type == "plan_approval":
                        available_actions = ["approve", "reject", "modify"]

                    out.append(
                        (
                            "hitl_interrupt",
                            {
                                "interrupt": interrupt_payload,
                                "interrupt_type": interrupt_type,
                                "data": interrupt_payload,
                                "available_actions": available_actions,
                            },
                        )
                    )

        return out


def _coerce_steps(value: object) -> list[EDAStep]:
    if value is None:
        return []
    if isinstance(value, list):
        steps: list[EDAStep] = []
        for item in value:
            if isinstance(item, EDAStep):
                steps.append(item)
            elif isinstance(item, dict):
                steps.append(EDAStep.model_validate(item))
        return steps
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return _coerce_steps(list(value))
    return []


def _coerce_cells(value: object) -> list[NotebookCell]:
    if value is None:
        return []
    if isinstance(value, list):
        cells: list[NotebookCell] = []
        for item in value:
            if isinstance(item, NotebookCell):
                cells.append(item)
            elif isinstance(item, dict):
                cells.append(NotebookCell.model_validate(item))
        return cells
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes, dict)):
        return _coerce_cells(list(value))
    return []
