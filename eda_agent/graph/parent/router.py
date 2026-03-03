"""Routing helpers for the parent graph.

Conditional edges live here to keep the Supervisor readable.
"""

from __future__ import annotations

from eda_agent.graph.parent.state import EDAState


def route_next(state: EDAState) -> str:
    if state.get("final_notebook_path"):
        return "__end__"
    eda_plan = list(state.get("eda_plan", []))
    idx = int(state.get("current_step_index", 0))
    if eda_plan and idx < len(eda_plan):
        return "eda_loop"
    if state.get("notebook_cells"):
        return "assembler"
    return "__end__"
