"""Parent graph (Supervisor) definition."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph

from eda_agent.graph.parent.state import EDAState
from eda_agent.graph.subgraphs.planner.graph import build_planner_subgraph


def build_supervisor_graph(
    *args: Any, checkpointer: BaseCheckpointSaver | None = None, **kwargs: Any
) -> Any:
    planner = build_planner_subgraph()
    graph = StateGraph(EDAState)

    def run_planner(state: EDAState) -> EDAState:
        out: Any = planner.invoke(
            {
                "messages": state.get("messages", []),
                "dataset_context": state.get("dataset_context"),
                "eda_plan": state.get("eda_plan", []),
            }
        )
        return {"eda_plan": out.get("eda_plan", []), "messages": out.get("messages", [])}

    graph.add_node("planner", run_planner)
    graph.set_entry_point("planner")
    graph.add_edge("planner", END)

    return graph.compile(checkpointer=checkpointer)
