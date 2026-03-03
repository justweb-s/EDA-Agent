from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from eda_agent.graph.subgraphs.critic.graph import build_critic_subgraph
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext
from eda_agent.models.notebook import NotebookCell


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


def test_critic_subgraph_returns_feedback(tmp_path: Path) -> None:
    critic = build_critic_subgraph()

    ctx = _fake_dataset_context(tmp_path)
    cells = [
        NotebookCell(
            cell_type="markdown",
            source="## Univariate: Numeric distributions\n\nSome notes\n",
            step_id="univariate_numeric",
            generated_at=datetime.now(UTC),
            re_executable=True,
        ),
        NotebookCell(
            cell_type="code",
            source="print('ok')\n",
            outputs=[],
            step_id="univariate_numeric",
            generated_at=datetime.now(UTC),
            re_executable=False,
        ),
    ]

    out = critic.invoke(
        {
            "notebook_cells": cells,
            "dataset_context": ctx,
            "section_name": "all",
        }
    )

    feedback = out.get("critic_feedback")
    assert feedback is not None
    assert feedback.verdict in {"ok", "needs_improvement"}
