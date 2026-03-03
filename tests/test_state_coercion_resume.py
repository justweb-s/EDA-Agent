from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import MemorySaver

from eda_agent.graph.parent.supervisor import build_supervisor_graph
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext
from eda_agent.models.execution import ExecutionResult, ExecutionSummary
from eda_agent.models.notebook import CellOutput, NotebookCell
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata


class _FakeRunResult:
    def __init__(self) -> None:
        self.execution_count = 1
        self.cell_outputs = [CellOutput(output_type="stream", text="ok")]
        self.execution = ExecutionResult(stdout="ok", stderr="", success=True, outputs=[])


class _FakeKernel:
    def execute(self, code: str, *, timeout_s: int, max_output_mb: float) -> Any:
        return _FakeRunResult()


def _fake_dataset_context(tmp_path: Path) -> DatasetContext:
    file_path = (tmp_path / "data.csv").resolve()
    file_path.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    return DatasetContext(
        file_path=str(file_path),
        file_name="data.csv",
        shape=(2, 2),
        columns=[
            ColumnInfo(
                name="a",
                dtype="int64",
                n_null=0,
                n_unique=2,
                sample_values=["1", "3"],
                detected_semantic_type="numeric",
            ),
            ColumnInfo(
                name="b",
                dtype="int64",
                n_null=0,
                n_unique=2,
                sample_values=["2", "4"],
                detected_semantic_type="numeric",
            ),
        ],
        memory_usage_mb=0.01,
        preview_markdown="|a|b|\n|---:|---:|\n|1|2|\n",
        basic_stats=BasicStats(numeric={}, categorical={}),
        detected_issues=[],
        created_at=datetime.now(UTC),
    )


def test_supervisor_accepts_dict_state_from_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HITL_DEFAULT_MODE", "none")

    checkpointer = MemorySaver()
    supervisor = build_supervisor_graph(checkpointer=checkpointer)

    session_id = "s-coercion"
    config = {"configurable": {"thread_id": session_id, "kernel": _FakeKernel()}}

    ctx = _fake_dataset_context(tmp_path)
    meta = SessionMetadata(
        session_id=session_id,
        started_at=datetime.now(UTC),
        llm_provider="mock",
        llm_model="mock",
        file_name=ctx.file_name,
    )

    plan = [
        EDAStep(
            step_id="data_quality",
            section="Data quality",
            title="Data quality checks",
            description="Review missing values, duplicates, and detected issues.",
            analysis_type="data_quality",
            target_columns=[],
            depends_on=[],
            is_mandatory=True,
            priority=10,
        ).model_dump(mode="json"),
        EDAStep(
            step_id="univariate_numeric",
            section="Univariate",
            title="Numeric distributions",
            description="Inspect distributions for numeric columns.",
            analysis_type="univariate",
            target_columns=["a", "b"],
            depends_on=["data_quality"],
            is_mandatory=True,
            priority=8,
        ).model_dump(mode="json"),
    ]

    seed_cell = NotebookCell(
        cell_type="markdown",
        source="# Seed",
        generated_at=datetime.now(UTC),
        re_executable=True,
    ).model_dump(mode="json")

    seed_history = ExecutionSummary(
        step_id="seed",
        section="Seed",
        findings="seed",
        columns_analyzed=[],
        created_at=datetime.now(UTC),
    ).model_dump(mode="json")

    resume_config = supervisor.update_state(
        config,
        {
            "dataset_context": ctx.model_dump(mode="json"),
            "messages": [],
            "hitl_enabled": False,
            "session_metadata": meta.model_dump(mode="json"),
            "eda_plan": plan,
            "current_step_index": 0,
            "notebook_cells": [seed_cell],
            "execution_history": [seed_history],
        },
        as_node="plan_approval",
    )

    out = supervisor.invoke(None, resume_config)

    notebook_path = str(out.get("final_notebook_path") or "")
    assert notebook_path

    path = Path(notebook_path)
    assert path.exists()
    assert path.stat().st_size > 50
