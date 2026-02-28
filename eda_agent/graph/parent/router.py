"""Routing helpers for the parent graph.

Conditional edges live here to keep the Supervisor readable.
"""

from __future__ import annotations

from eda_agent.graph.parent.state import EDAState


def route_next(state: EDAState) -> str:
    raise NotImplementedError
