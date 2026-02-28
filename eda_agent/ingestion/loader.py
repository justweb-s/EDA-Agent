"""Dataset loading utilities."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from eda_agent.exceptions import DataIngestionError


def load_file(file_path: str | Path) -> pd.DataFrame:
    """Load a dataset file into a Pandas DataFrame.

    Supported formats:
    - CSV (.csv)
    - Excel (.xls, .xlsx)
    """

    path = Path(file_path)
    if not path.exists():
        raise DataIngestionError(f"File not found: {path}")

    suffix = path.suffix.lower()

    try:
        if suffix == ".csv":
            return pd.read_csv(path)
        if suffix in {".xls", ".xlsx"}:
            return pd.read_excel(path)
    except Exception as exc:  # noqa: BLE001
        raise DataIngestionError(f"Failed to load file: {path}") from exc

    raise DataIngestionError(f"Unsupported file type: {suffix}")
