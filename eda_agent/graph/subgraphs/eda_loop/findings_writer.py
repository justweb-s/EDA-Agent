"""FINDINGS_WRITER node."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from eda_agent.graph.coercion import (
    coerce_cells,
    coerce_execution_history,
    coerce_execution_result,
    coerce_step,
)
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.models.execution import ExecutionSummary, ObserverVerdict
from eda_agent.models.notebook import NotebookCell


def findings_writer_node(state: EDALoopState, *_: Any, **__: Any) -> Any:

    step = coerce_step(state.get("current_step"))
    execution = coerce_execution_result(state.get("execution_result"))
    if step is None or execution is None:
        return {
            "current_step_index": int(state.get("current_step_index", 0)) + 1,
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

    verdict_raw = state.get("observer_verdict_obj")
    verdict_obj: ObserverVerdict | None = None
    if isinstance(verdict_raw, ObserverVerdict):
        verdict_obj = verdict_raw
    elif isinstance(verdict_raw, dict):
        try:
            verdict_obj = ObserverVerdict.model_validate(verdict_raw)
        except Exception:
            verdict_obj = None

    findings = ""
    if verdict_obj is not None and verdict_obj.findings_description:
        findings = verdict_obj.findings_description
    else:
        stdout = (execution.stdout or "").strip()
        stderr = (execution.stderr or "").strip()
        if stdout:
            findings = stdout.splitlines()[0]
        elif stderr:
            findings = stderr.splitlines()[0]
        else:
            findings = "Step executed."

    execution_history = coerce_execution_history(state.get("execution_history", []))
    summary = ExecutionSummary(
        step_id=step.step_id,
        section=step.section,
        findings=findings,
        key_statistics=(verdict_obj.key_statistics if verdict_obj is not None else {}),
        charts_produced=(verdict_obj.charts_produced if verdict_obj is not None else []),
        anomalies_found=(verdict_obj.anomalies_found if verdict_obj is not None else []),
        columns_analyzed=list(step.target_columns or []),
        created_at=datetime.now(UTC),
    )
    execution_history.append(summary)

    return {
        "notebook_cells": notebook_cells,
        "execution_history": execution_history,
        "execution_summary": summary,
        "current_step_index": int(state.get("current_step_index", 0)) + 1,
        "local_retry_count": 0,
        "retry_messages": [],
    }
