"""Routing helpers for the parent graph.

Conditional edges live here to keep the Supervisor readable.
"""

from __future__ import annotations

from eda_agent.graph.parent.state import EDAState


def route_next(state: EDAState) -> str:
    if state.get("final_notebook_path"):
        return "__end__"
    if state.get("notebook_cells"):
        return "assembler"
    return "__end__"
