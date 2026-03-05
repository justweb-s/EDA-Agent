"""Parent graph (Supervisor) definition."""

from __future__ import annotations

from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph
from langgraph.types import interrupt

from eda_agent.graph.coercion import (
    coerce_cells,
    coerce_dataset_context,
    coerce_execution_history,
    coerce_plan,
)
from eda_agent.graph.parent.assembler import assemble_notebook
from eda_agent.graph.parent.router import route_next
from eda_agent.graph.parent.state import EDAState
from eda_agent.graph.subgraphs.critic.graph import build_critic_subgraph
from eda_agent.graph.subgraphs.eda_loop.graph import build_eda_loop_subgraph
from eda_agent.graph.subgraphs.planner.graph import build_planner_subgraph


def build_supervisor_graph(
    *args: Any, checkpointer: BaseCheckpointSaver | None = None, **kwargs: Any
) -> Any:
    planner = build_planner_subgraph()
    eda_loop = build_eda_loop_subgraph()
    critic = build_critic_subgraph()
    graph = StateGraph(EDAState)

    def run_critic(state: EDAState, config: RunnableConfig) -> dict[str, Any]:
        dataset_context = coerce_dataset_context(state.get("dataset_context"))
        notebook_cells = coerce_cells(state.get("notebook_cells", []))
        out: Any = critic.invoke(
            {
                "notebook_cells": notebook_cells,
                "dataset_context": dataset_context,
                "section_name": "all",
            },
            config,
        )
        return {"critic_feedback": out.get("critic_feedback")}

    def run_eda_loop(state: EDAState, config: RunnableConfig) -> dict[str, Any]:
        dataset_context = coerce_dataset_context(state.get("dataset_context"))
        eda_plan = coerce_plan(state.get("eda_plan", []))
        notebook_cells = coerce_cells(state.get("notebook_cells", []))
        execution_history = coerce_execution_history(state.get("execution_history", []))
        out: Any = eda_loop.invoke(
            {
                "eda_plan": eda_plan,
                "current_step_index": int(state.get("current_step_index", 0)),
                "dataset_context": dataset_context,
                "session_metadata": state.get("session_metadata"),
                "notebook_cells": notebook_cells,
                "execution_history": execution_history,
            },
            config,
        )
        return {
            "current_step_index": int(out.get("current_step_index", 0)),
            "notebook_cells": out.get("notebook_cells", []),
            "execution_history": out.get("execution_history", []),
        }

    def run_planner(state: EDAState) -> EDAState:
        dataset_context = coerce_dataset_context(state.get("dataset_context"))
        eda_plan = coerce_plan(state.get("eda_plan", []))
        out: Any = planner.invoke(
            {
                "messages": state.get("messages", []),
                "dataset_context": dataset_context,
                "eda_plan": eda_plan,
                "session_metadata": state.get("session_metadata"),
            }
        )
        return {
            **state,
            "eda_plan": out.get("eda_plan", []),
            "messages": out.get("messages", state.get("messages", [])),
        }

    def plan_approval(state: EDAState) -> dict[str, Any]:
        if not state.get("hitl_enabled", False):
            return dict(state)

        eda_plan = coerce_plan(state.get("eda_plan", []))

        resume_value = interrupt(
            {
                "type": "plan_approval",
                "eda_plan": [s.model_dump(mode="json") for s in eda_plan],
            }
        )
        return {**state, "hitl_plan_approval": resume_value}

    graph.add_node("planner", run_planner)
    graph.add_node("plan_approval", plan_approval)
    graph.add_node("eda_loop", run_eda_loop)
    graph.add_node("critic", run_critic)
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
    graph.add_edge("eda_loop", "critic")
    graph.add_edge("critic", "assembler")
    graph.add_edge("assembler", END)

    return graph.compile(checkpointer=checkpointer)
