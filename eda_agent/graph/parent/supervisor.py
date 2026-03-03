"""Parent graph (Supervisor) definition."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from eda_agent.graph.parent.assembler import assemble_notebook
from eda_agent.graph.parent.router import route_next
from eda_agent.graph.parent.state import EDAState
from eda_agent.graph.subgraphs.eda_loop.graph import build_eda_loop_subgraph
from eda_agent.graph.subgraphs.planner.graph import build_planner_subgraph


def build_supervisor_graph(
    *args: Any, checkpointer: BaseCheckpointSaver | None = None, **kwargs: Any
) -> Any:
    planner = build_planner_subgraph()
    eda_loop = build_eda_loop_subgraph()
    graph = StateGraph(EDAState)

    def run_eda_loop(state: EDAState, config: RunnableConfig) -> dict[str, Any]:
        out: Any = eda_loop.invoke(
            {
                "eda_plan": state.get("eda_plan", []),
                "current_step_index": int(state.get("current_step_index", 0)),
                "dataset_context": state.get("dataset_context"),
                "notebook_cells": state.get("notebook_cells", []),
                "execution_history": state.get("execution_history", []),
            },
            config,
        )
        return {
            "current_step_index": int(out.get("current_step_index", 0)),
            "notebook_cells": out.get("notebook_cells", []),
            "execution_history": out.get("execution_history", []),
        }

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
    graph.add_node("eda_loop", run_eda_loop)
    graph.add_node("assembler", assemble_notebook)
    graph.set_entry_point("planner")
    graph.add_edge("planner", "plan_approval")
    graph.add_conditional_edges(
        "plan_approval",
        route_next,
        {
            "eda_loop": "eda_loop",
            "assembler": "assembler",
            "__end__": END,
        },
    )
    graph.add_edge("eda_loop", "assembler")
    graph.add_edge("assembler", END)

    return graph.compile(checkpointer=checkpointer)
