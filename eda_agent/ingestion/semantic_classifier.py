"""Column semantic classification heuristics."""

from __future__ import annotations

import re
from typing import Literal

import pandas as pd

SemanticType = Literal[
    "numeric_continuous",
    "numeric_discrete",
    "categorical_low_cardinality",
    "categorical_high_cardinality",
    "datetime",
    "text_free",
    "boolean",
    "id_column",
    "target_variable",
    "unknown",
]


_TARGET_RE = re.compile(r"\b(target|label|outcome|y)\b", re.IGNORECASE)
_ID_RE = re.compile(r"\b(id|uuid|guid)\b", re.IGNORECASE)


def classify_column(
    name: str,
    series: pd.Series,
    *,
    high_cardinality_threshold: int,
) -> SemanticType:
    """Classify a column using simple heuristics based on name and values."""

    clean_name = name.strip()

    non_null = series.dropna()
    n_unique = int(non_null.nunique(dropna=True))

    if _TARGET_RE.search(clean_name):
        return "target_variable"

    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"

    if pd.api.types.is_bool_dtype(series):
        return "boolean"

    if n_unique == 2:
        return "boolean"

    if _ID_RE.search(clean_name) and n_unique == len(non_null) and len(non_null) > 0:
        return "id_column"

    if pd.api.types.is_numeric_dtype(series):
        if pd.api.types.is_integer_dtype(series) and n_unique <= high_cardinality_threshold:
            return "numeric_discrete"
        return "numeric_continuous"

    if pd.api.types.is_string_dtype(series) or pd.api.types.is_object_dtype(series):
        if len(non_null) == 0:
            return "unknown"

        avg_len = float(non_null.astype(str).str.len().mean())
        if avg_len >= 30 and n_unique >= high_cardinality_threshold:
            return "text_free"

        if n_unique < high_cardinality_threshold:
            return "categorical_low_cardinality"
        return "categorical_high_cardinality"

    return "unknown"
