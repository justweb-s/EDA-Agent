"""Automatic dataset issue detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd


@dataclass(frozen=True)
class DetectedIssue:
    """Represents a detected issue during ingestion/profiling."""

    code: str
    message: str
    severity: Literal["warning", "error"]
    column: str | None = None


def detect_issues(
    df: pd.DataFrame,
    *,
    null_warning_threshold: float,
    null_error_threshold: float,
) -> list[DetectedIssue]:
    """Detect common issues using lightweight heuristics."""

    issues: list[DetectedIssue] = []

    n_rows = len(df)
    if n_rows == 0:
        issues.append(
            DetectedIssue(code="EMPTY_DATASET", message="Dataset has 0 rows", severity="error")
        )
        return issues

    for col in df.columns:
        series = df[col]
        null_pct = float(series.isna().mean())

        if null_pct >= null_error_threshold:
            issues.append(
                DetectedIssue(
                    code="HIGH_NULLS",
                    message=f"Column '{col}' has {null_pct:.1%} null values",
                    severity="error",
                    column=str(col),
                )
            )
        elif null_pct >= null_warning_threshold:
            issues.append(
                DetectedIssue(
                    code="NULLS",
                    message=f"Column '{col}' has {null_pct:.1%} null values",
                    severity="warning",
                    column=str(col),
                )
            )

        n_unique = int(series.nunique(dropna=True))
        if n_unique == 1:
            issues.append(
                DetectedIssue(
                    code="CONSTANT_COLUMN",
                    message=f"Column '{col}' is constant",
                    severity="warning",
                    column=str(col),
                )
            )

    duplicated_rows = int(df.duplicated().sum())
    if duplicated_rows > 0:
        issues.append(
            DetectedIssue(
                code="DUPLICATE_ROWS",
                message=f"Dataset contains {duplicated_rows} duplicated rows",
                severity="warning",
            )
        )

    return issues
