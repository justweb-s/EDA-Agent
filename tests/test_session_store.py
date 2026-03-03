from __future__ import annotations

from pathlib import Path

from dataclasses import replace
from datetime import UTC, datetime

from eda_agent.api.session_store import SessionStore, new_session_record
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext
from eda_agent.models.notebook import NotebookCell


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


def test_session_store_roundtrip(tmp_path: Path) -> None:
    store = SessionStore(output_dir=(tmp_path / "outputs"))

    ctx = _fake_dataset_context(tmp_path)
    record = new_session_record(
        session_id="s1",
        file_name=ctx.file_name,
        file_path=ctx.file_path,
        dataset_context=ctx,
    )

    store.upsert(record)
    got = store.get("s1")
    assert got is not None
    assert got.session_id == "s1"
    assert got.status == "created"
    assert got.dataset_context.file_name == "data.csv"

    cell = NotebookCell(
        cell_type="markdown",
        source="# Title",
        generated_at=datetime.now(UTC),
        re_executable=True,
    )
    store.upsert(
        replace(
            got,
            status="completed",
            n_cells=1,
            notebook_cells=[cell],
            notebook_path="some/path.ipynb",
        )
    )

    got2 = store.get("s1")
    assert got2 is not None
    assert got2.status == "completed"
    assert got2.n_cells == 1
    assert got2.notebook_cells is not None
    assert len(got2.notebook_cells) == 1
    assert got2.notebook_cells[0].cell_type == "markdown"
    assert got2.notebook_path == "some/path.ipynb"

    listing = store.list()
    assert any(r.session_id == "s1" for r in listing)

    assert store.delete("s1") is True
    assert store.get("s1") is None
    assert store.delete("s1") is False
