"""EDA Loop subgraph definition."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from langgraph.graph import END, StateGraph

from eda_agent.graph.subgraphs.eda_loop.coder import coder_node
from eda_agent.graph.subgraphs.eda_loop.executor import executor_node
from eda_agent.graph.subgraphs.eda_loop.findings_writer import findings_writer_node
from eda_agent.graph.subgraphs.eda_loop.observer import observer_node
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep


def build_eda_loop_subgraph(*args: Any, **kwargs: Any) -> Any:
    graph = StateGraph(EDALoopState)

    def select_step(state: EDALoopState) -> EDALoopState:
        eda_plan_raw = list(state.get("eda_plan", []))
        eda_plan: list[EDAStep] = [
            (s if isinstance(s, EDAStep) else EDAStep.model_validate(s)) for s in eda_plan_raw
        ]
        current_step_index = int(state.get("current_step_index", 0))

        notebook_cells_raw = list(state.get("notebook_cells", []))
        notebook_cells: list[NotebookCell] = [
            (c if isinstance(c, NotebookCell) else NotebookCell.model_validate(c))
            for c in notebook_cells_raw
        ]
        if eda_plan and current_step_index == 0:
            has_plan_cell = any(
                (c.cell_type == "markdown") and ("## Plan" in c.source) for c in notebook_cells
            )
            if not has_plan_cell:
                plan_md = "## Plan\n\n"
                for i, step in enumerate(eda_plan, start=1):
                    cols = (
                        ", ".join(f"`{c}`" for c in step.target_columns)
                        if step.target_columns
                        else ""
                    )
                    cols_line = f"\\n  - **Columns**: {cols}" if cols else ""
                    plan_md += (
                        f"{i}. **{step.section}** — {step.title}\\n"
                        f"  - {step.description}{cols_line}\\n"
                    )

                notebook_cells.append(
                    NotebookCell(
                        cell_type="markdown",
                        source=plan_md,
                        generated_at=datetime.now(UTC),
                        re_executable=True,
                    )
                )

        if current_step_index >= len(eda_plan):
            return {
                **state,
                "current_step_index": current_step_index,
                "notebook_cells": notebook_cells,
            }

        current_step = eda_plan[current_step_index]
        return {
            **state,
            "current_step_index": current_step_index,
            "current_step": current_step,
            "notebook_cells": notebook_cells,
        }

    def route_after_select(state: EDALoopState) -> str:
        eda_plan = list(state.get("eda_plan", []))
        idx = int(state.get("current_step_index", 0))
        if idx >= len(eda_plan):
            return "__end__"
        return "coder"

    def route_after_observer(state: EDALoopState) -> str:
        if state.get("observer_verdict") == "retry":
            return "coder"
        return "findings_writer"

    graph.add_node("select_step", select_step)
    graph.add_node("coder", coder_node)
    graph.add_node("executor", executor_node)
    graph.add_node("observer", observer_node)
    graph.add_node("findings_writer", findings_writer_node)

    graph.set_entry_point("select_step")
    graph.add_conditional_edges(
        "select_step",
        route_after_select,
        {
            "coder": "coder",
            "__end__": END,
        },
    )
    graph.add_edge("coder", "executor")
    graph.add_edge("executor", "observer")
    graph.add_conditional_edges(
        "observer",
        route_after_observer,
        {
            "coder": "coder",
            "findings_writer": "findings_writer",
        },
    )

    graph.add_edge("findings_writer", "select_step")

    return graph.compile()
