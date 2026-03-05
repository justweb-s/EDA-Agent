from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import MemorySaver

from eda_agent.graph.parent.supervisor import build_supervisor_graph
from eda_agent.graph.subgraphs.eda_loop.graph import build_eda_loop_subgraph
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata


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


def test_supervisor_runs_eda_loop_and_assembles_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("HITL_DEFAULT_MODE", "none")

    checkpointer = MemorySaver()
    supervisor = build_supervisor_graph(checkpointer=checkpointer)

    session_id = "s-eda-loop"
    config = {"configurable": {"thread_id": session_id}}

    ctx = _fake_dataset_context(tmp_path)

    meta = SessionMetadata(
        session_id=session_id,
        started_at=datetime.now(UTC),
        llm_provider="mock",
        llm_model="mock",
        file_name=ctx.file_name,
    )

    out = supervisor.invoke(
        {
            "dataset_context": ctx,
            "messages": [],
            "hitl_enabled": False,
            "session_metadata": meta,
        },
        config,
    )

    notebook_cells = list(out.get("notebook_cells", []))
    assert notebook_cells
    assert any(
        (getattr(c, "cell_type", None) == "markdown")
        and ("## Plan" in str(getattr(c, "source", "")))
        for c in notebook_cells
    )

    notebook_path = str(out.get("final_notebook_path") or "")
    assert notebook_path

    path = Path(notebook_path)
    assert path.exists()
    assert path.stat().st_size > 100

    expected = ((tmp_path / "outputs") / "notebooks" / f"eda-agent-{session_id}.ipynb").resolve()
    assert path.resolve() == expected


def test_eda_loop_retries_then_succeeds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ctx = _fake_dataset_context(tmp_path)

    plan = [
        EDAStep(
            step_id="dq",
            section="Data quality",
            title="Check nulls",
            description="Check missing values.",
            analysis_type="data_quality",
            target_columns=[],
            depends_on=[],
            is_mandatory=True,
            priority=10,
        )
    ]

    class KernelStub:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, *_: object, **__: object) -> object:
            self.calls += 1
            if self.calls == 1:
                return type(
                    "RunResult",
                    (),
                    {
                        "execution_count": 1,
                        "cell_outputs": [CellOutput(output_type="error", text="NameError")],
                        "execution": ExecutionResult(
                            stdout="",
                            stderr="Traceback: NameError",
                            success=False,
                            outputs=[],
                        ),
                    },
                )()

            return type(
                "RunResult",
                (),
                {
                    "execution_count": 2,
                    "cell_outputs": [CellOutput(output_type="stream", text="ok")],
                    "execution": ExecutionResult(
                        stdout="ok\n",
                        stderr="",
                        success=True,
                        outputs=[],
                    ),
                },
            )()

    kernel = KernelStub()
    config = {"configurable": {"thread_id": "t-retry", "kernel": kernel}}

    monkeypatch.setenv(
        "MOCK_LLM_RESPONSES",
        "["
        '{"code":"df.head()","expected_output_description":"preview"},'
        '{"verdict":"retry","findings_description":"error","retry_hint":"fix"},'
        '{"code":"df.head()","expected_output_description":"preview"},'
        '{"verdict":"success","findings_description":"ok"}'
        "]",
    )

    graph = build_eda_loop_subgraph()
    out = graph.invoke(
        {
            "eda_plan": plan,
            "current_step_index": 0,
            "dataset_context": ctx,
            "notebook_cells": [],
            "execution_history": [],
            "retry_messages": [],
            "local_retry_count": 0,
        },
        config,
    )

    assert kernel.calls == 2
    assert int(out.get("current_step_index", -1)) == 1
    assert len(list(out.get("execution_history", []))) == 1
