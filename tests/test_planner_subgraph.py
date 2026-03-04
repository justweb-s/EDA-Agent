from __future__ import annotations

import json
from datetime import UTC, datetime

import pytest

from eda_agent.graph.subgraphs.planner.graph import build_planner_subgraph
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext


@pytest.mark.asyncio
async def test_planner_validate_loop_regenerates_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    responses = [
        "Dataset analysis",
        {
            "steps": [
                {
                    "step_id": "data_quality",
                    "section": "Data quality",
                    "title": "Data quality checks",
                    "description": "...",
                    "analysis_type": "data_quality",
                    "target_columns": [],
                    "depends_on": [],
                    "is_mandatory": True,
                    "priority": 10,
                },
                {
                    "step_id": "data_quality",
                    "section": "Univariate",
                    "title": "Numeric",
                    "description": "...",
                    "analysis_type": "univariate",
                    "target_columns": [],
                    "depends_on": ["data_quality"],
                    "is_mandatory": True,
                    "priority": 8,
                },
            ]
        },
        {
            "steps": [
                {
                    "step_id": "data_quality",
                    "section": "Data quality",
                    "title": "Data quality checks",
                    "description": "...",
                    "analysis_type": "data_quality",
                    "target_columns": [],
                    "depends_on": [],
                    "is_mandatory": True,
                    "priority": 10,
                },
                {
                    "step_id": "univariate_numeric",
                    "section": "Univariate",
                    "title": "Numeric",
                    "description": "...",
                    "analysis_type": "univariate",
                    "target_columns": [],
                    "depends_on": ["data_quality"],
                    "is_mandatory": True,
                    "priority": 8,
                },
            ]
        },
    ]
    monkeypatch.setenv("MOCK_LLM_RESPONSES", json.dumps(responses))

    dataset_context = DatasetContext(
        file_path="/tmp/data.csv",
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
                sample_values=["1", "2", "3"],
                detected_semantic_type="numeric",
            ),
        ],
        memory_usage_mb=0.001,
        preview_markdown="a | b\n1 | 1\n",
        basic_stats=BasicStats(numeric={}, categorical={}),
        detected_issues=[],
        created_at=datetime.now(UTC),
    )

    planner = build_planner_subgraph()
    out = planner.invoke({"dataset_context": dataset_context, "messages": []})

    assert out.get("eda_plan")
    step_ids = [s.step_id for s in out["eda_plan"]]
    assert len(step_ids) == len(set(step_ids))
    assert out.get("validation_errors", []) == []
