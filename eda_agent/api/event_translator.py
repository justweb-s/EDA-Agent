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
            mode, payload = event
            if mode == "updates":
                event = payload

        if not isinstance(event, dict):
            return []

        out: list[tuple[str, dict]] = []
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
                    out.append(("cell_added", {"cell": cell.model_dump(mode="json")}))

            if "final_notebook_path" in update and update.get("final_notebook_path"):
                out.append(
                    (
                        "analysis_completed",
                        {"notebook_path": str(update.get("final_notebook_path"))},
                    )
                )

            if "__interrupt__" in update:
                interrupt_payload = update.get("__interrupt__")
                if interrupt_payload is not None:
                    out.append(("hitl_interrupt", {"interrupt": interrupt_payload}))

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
