"""Parent graph (Supervisor) definition."""

from __future__ import annotations

from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

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
        return {
            **state,
            "eda_plan": out.get("eda_plan", []),
            "messages": out.get("messages", []),
        }

    def plan_approval(state: EDAState) -> dict[str, Any]:
        if not state.get("hitl_enabled", False):
            return dict(state)

        resume_value = interrupt(
            {
                "type": "plan_approval",
                "eda_plan": [
                    (s.model_dump(mode="json") if hasattr(s, "model_dump") else s)
                    for s in state.get("eda_plan", [])
                ],
            }
        )
        return {**state, "hitl_plan_approval": resume_value}

    graph.add_node("planner", run_planner)
    graph.add_node("plan_approval", plan_approval)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "plan_approval")
    graph.add_edge("plan_approval", END)

    return graph.compile(checkpointer=checkpointer)
