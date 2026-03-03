from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from eda_agent.graph.parent.supervisor import build_supervisor_graph
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.session import SessionMetadata


def _fake_dataset_context(tmp_path: Path) -> DatasetContext:
    file_path = str((tmp_path / "data.csv").resolve())
    return DatasetContext(
        file_path=file_path,
        file_name="data.csv",
        shape=(3, 2),
        columns=[
            ColumnInfo(
                name="a",
                dtype="int64",
                n_null=0,
                n_unique=3,
                sample_values=["1", "2", "3"],
                detected_semantic_type="numeric",
            ),
            ColumnInfo(
                name="b",
                dtype="int64",
                n_null=0,
                n_unique=3,
                sample_values=["2", "4", "6"],
                detected_semantic_type="numeric",
            ),
        ],
        memory_usage_mb=0.01,
        preview_markdown="|a|b|\n|---:|---:|\n|1|2|\n",
        basic_stats=BasicStats(numeric={}, categorical={}),
        detected_issues=[],
        created_at=datetime.now(UTC),
    )


def test_supervisor_assembler_writes_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))

    checkpointer = MemorySaver()
    supervisor = build_supervisor_graph(checkpointer=checkpointer)

    session_id = "s-assembler"
    config = {"configurable": {"thread_id": session_id}}

    ctx = _fake_dataset_context(tmp_path)
    meta = SessionMetadata(
        session_id=session_id,
        started_at=datetime.now(UTC),
        llm_provider="mock",
        llm_model="mock",
        file_name=ctx.file_name,
    )

    supervisor.invoke(
        {
            "dataset_context": ctx,
            "messages": [],
            "hitl_enabled": False,
            "session_metadata": meta,
        },
        config,
    )

    cells = [
        NotebookCell(
            cell_type="markdown",
            source="# Hello",
            generated_at=datetime.now(UTC),
            re_executable=True,
        )
    ]

    assembler_config = supervisor.update_state(
        config,
        {"notebook_cells": cells},
        as_node="plan_approval",
    )

    out = supervisor.invoke(None, assembler_config)
    notebook_path = str(out.get("final_notebook_path") or "")
    assert notebook_path

    path = Path(notebook_path)
    assert path.exists()
    assert path.stat().st_size > 50

    expected = ((tmp_path / "outputs") / "notebooks" / f"eda-agent-{session_id}.ipynb").resolve()
    assert path.resolve() == expected
