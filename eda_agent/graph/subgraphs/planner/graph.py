"""Planner subgraph definition."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from eda_agent.config import EDAConfig
from eda_agent.graph.subgraphs.planner.nodes import analyze_dataset, generate_plan, validate_plan
from eda_agent.graph.subgraphs.planner.state import PlannerState


def build_planner_subgraph(*args: Any, **kwargs: Any) -> Any:
    graph = StateGraph(PlannerState)

    def route_after_validate(state: PlannerState) -> str:
        errors = list(state.get("validation_errors", []))
        if not errors:
            return "finalize"

        cfg = EDAConfig()
        iterations = int(state.get("iteration_count", 0))
        if iterations < max(1, int(cfg.llm_max_retries)):
            return "generate"
        return "finalize"

    def finalize(state: PlannerState) -> PlannerState:
        errors = list(state.get("validation_errors", []))
        if state.get("eda_plan"):
            plan = list(state.get("eda_plan", []))
        else:
            plan = list(state.get("draft_plan", []))
            if not plan and not errors:
                errors.append("Planner produced an empty plan")
        return {
            "eda_plan": plan,
            "validation_errors": errors,
            "iteration_count": int(state.get("iteration_count", 0)),
            "messages": list(state.get("messages", [])),
        }

    graph.add_node("analyze", analyze_dataset)
    graph.add_node("generate", generate_plan)
    graph.add_node("validate", validate_plan)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("analyze")
    graph.add_edge("analyze", "generate")
    graph.add_edge("generate", "validate")
    graph.add_conditional_edges(
        "validate",
        route_after_validate,
        {
            "generate": "generate",
            "finalize": "finalize",
        },
    )
    graph.add_edge("finalize", END)

    return graph.compile()
