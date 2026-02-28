"""Dataset profiling and `DatasetContext` construction."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from eda_agent.config import EDAConfig
from eda_agent.ingestion.issue_detector import detect_issues
from eda_agent.ingestion.semantic_classifier import classify_column
from eda_agent.models.dataset import BasicStats, ColumnInfo, DatasetContext, DetectedIssueModel


def build_dataset_context(
    df: pd.DataFrame,
    *,
    file_path: str | Path,
    config: EDAConfig,
) -> DatasetContext:
    """Build an immutable dataset context used as foundational prompt input."""

    path = Path(file_path)
    memory_usage_mb = float(df.memory_usage(deep=True).sum() / (1024 * 1024))

    preview_markdown = df.head(5).to_markdown(index=False)

    issues = detect_issues(
        df,
        null_warning_threshold=config.null_warning_threshold,
        null_error_threshold=config.null_error_threshold,
    )

    columns: list[ColumnInfo] = []
    basic_stats = BasicStats(numeric={}, categorical={})

    for col in df.columns:
        series = df[col]
        non_null = series.dropna()

        detected_semantic_type = classify_column(
            str(col),
            series,
            high_cardinality_threshold=config.high_cardinality_threshold,
        )

        sample_values = [str(v) for v in non_null.head(5).tolist()]

        column_info = ColumnInfo(
            name=str(col),
            dtype=str(series.dtype),
            n_null=int(series.isna().sum()),
            n_unique=int(non_null.nunique(dropna=True)),
            sample_values=sample_values,
            detected_semantic_type=detected_semantic_type,
        )
        columns.append(column_info)

        if pd.api.types.is_numeric_dtype(series):
            desc = non_null.describe(percentiles=[0.25, 0.5, 0.75]).to_dict()
            basic_stats.numeric[str(col)] = {k: _safe_number(v) for k, v in desc.items()}
        else:
            vc = non_null.astype(str).value_counts(dropna=True).head(10)
            basic_stats.categorical[str(col)] = {str(k): int(v) for k, v in vc.items()}

    detected_issue_models = [
        DetectedIssueModel(code=i.code, message=i.message, severity=i.severity, column=i.column)
        for i in issues
    ]

    return DatasetContext(
        file_path=str(path.resolve()),
        file_name=path.name,
        shape=(int(df.shape[0]), int(df.shape[1])),
        columns=columns,
        memory_usage_mb=memory_usage_mb,
        preview_markdown=preview_markdown,
        basic_stats=basic_stats,
        detected_issues=detected_issue_models,
        created_at=datetime.now(UTC),
    )


def _safe_number(value: object) -> float | int | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)  # type: ignore[arg-type]
    except Exception:  # noqa: BLE001
        return None
