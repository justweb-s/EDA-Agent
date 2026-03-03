"""Critic subgraph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from eda_agent.graph.subgraphs.critic.nodes import evaluate_section, read_section
from eda_agent.graph.subgraphs.critic.state import CriticState


def build_critic_subgraph(*args: Any, **kwargs: Any) -> Any:
    graph = StateGraph(CriticState)

    graph.add_node("read_section", read_section)
    graph.add_node("evaluate_section", evaluate_section)
    graph.set_entry_point("read_section")
    graph.add_edge("read_section", "evaluate_section")
    graph.add_edge("evaluate_section", END)

    return graph.compile()
