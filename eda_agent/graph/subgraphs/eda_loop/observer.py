"""OBSERVER node."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, cast

from eda_agent.config import EDAConfig
from eda_agent.graph.coercion import (
    coerce_cells,
    coerce_execution_history,
    coerce_execution_result,
    coerce_step,
)
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.models.execution import ExecutionSummary
from eda_agent.models.notebook import NotebookCell


def observer_node(*args: Any, **kwargs: Any) -> Any:
    state = cast(EDALoopState, args[0] if args else kwargs.get("state"))

    step = coerce_step(state.get("current_step"))
    execution = coerce_execution_result(state.get("execution_result"))
    if step is None or execution is None:
        return {
            "observer_verdict": "fatal_error",
            "retry_messages": ["Missing current_step or execution_result"],
        }

    cfg = EDAConfig()
    local_retry_count = int(state.get("local_retry_count", 0))
    retry_messages = list(state.get("retry_messages", []))

    if not execution.success:
        error_msg = execution.stderr or "Execution failed"
        retry_messages.append(error_msg)
        if local_retry_count < (cfg.max_step_retries - 1):
            return {
                "observer_verdict": "retry",
                "retry_messages": retry_messages,
                "local_retry_count": local_retry_count + 1,
            }

        notebook_cells = coerce_cells(state.get("notebook_cells", []))

        step_md = f"## {step.section}: {step.title}\n\n{step.description}\n"
        notebook_cells.append(
            NotebookCell(
                cell_type="markdown",
                source=step_md,
                generated_at=datetime.now(UTC),
                re_executable=True,
                step_id=step.step_id,
            )
        )

        notebook_cells.append(
            NotebookCell(
                cell_type="code",
                source=str(state.get("generated_code") or ""),
                outputs=list(state.get("cell_outputs", [])),
                execution_count=state.get("execution_count"),
                step_id=step.step_id,
                generated_at=datetime.now(UTC),
                re_executable=False,
            )
        )

        execution_history = coerce_execution_history(state.get("execution_history", []))
        execution_history.append(
            ExecutionSummary(
                step_id=step.step_id,
                section=step.section,
                findings=f"Step failed after retries: {error_msg}",
                columns_analyzed=list(step.target_columns or []),
                created_at=datetime.now(UTC),
            )
        )

        return {
            "observer_verdict": "fatal_error",
            "notebook_cells": notebook_cells,
            "execution_history": execution_history,
            "execution_summary": execution_history[-1],
            "current_step_index": int(state.get("current_step_index", 0)) + 1,
            "local_retry_count": 0,
            "retry_messages": [],
        }

    notebook_cells = coerce_cells(state.get("notebook_cells", []))

    step_md = f"## {step.section}: {step.title}\n\n{step.description}\n"
    if step.target_columns:
        cols = ", ".join(f"`{c}`" for c in step.target_columns)
        step_md += f"\n**Columns**: {cols}\n"

    notebook_cells.append(
        NotebookCell(
            cell_type="markdown",
            source=step_md,
            generated_at=datetime.now(UTC),
            re_executable=True,
            step_id=step.step_id,
        )
    )

    notebook_cells.append(
        NotebookCell(
            cell_type="code",
            source=str(state.get("generated_code") or ""),
            outputs=list(state.get("cell_outputs", [])),
            execution_count=state.get("execution_count"),
            step_id=step.step_id,
            generated_at=datetime.now(UTC),
            re_executable=False,
        )
    )

    execution_history = coerce_execution_history(state.get("execution_history", []))
    stdout = (execution.stdout or "").strip()
    findings = stdout.splitlines()[0] if stdout else "Step executed successfully."
    summary = ExecutionSummary(
        step_id=step.step_id,
        section=step.section,
        findings=findings,
        columns_analyzed=list(step.target_columns or []),
        created_at=datetime.now(UTC),
    )
    execution_history.append(summary)

    return {
        "observer_verdict": "success",
        "notebook_cells": notebook_cells,
        "execution_history": execution_history,
        "execution_summary": summary,
        "current_step_index": int(state.get("current_step_index", 0)) + 1,
        "local_retry_count": 0,
        "retry_messages": [],
    }
