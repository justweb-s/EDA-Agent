"""CODER node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from eda_agent.graph.coercion import coerce_dataset_context, coerce_step
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState


def coder_node(*args: Any, **kwargs: Any) -> Any:
    state = cast(EDALoopState, args[0] if args else kwargs.get("state"))

    step = coerce_step(state.get("current_step"))
    dataset_context = coerce_dataset_context(state.get("dataset_context"))
    if step is None or dataset_context is None:
        return {"generated_code": "raise ValueError('Missing current_step or dataset_context')"}

    file_path = str(dataset_context.file_path)
    suffix = Path(file_path).suffix.lower()
    read_stmt = (
        f'df = pd.read_excel(r"{file_path}")'
        if suffix in {".xls", ".xlsx"}
        else f'df = pd.read_csv(r"{file_path}")'
    )

    prelude = (
        "import pandas as pd\n"
        "\n"
        f'file_path = r"{file_path}"\n'
        "if 'df' not in globals():\n"
        f"    {read_stmt}\n"
    )

    cols = list(step.target_columns or [])
    cols_expr = repr(cols)

    if step.analysis_type == "data_quality":
        body = (
            "missing = df.isna().mean().sort_values(ascending=False).head(15)\n"
            "duplicates = int(df.duplicated().sum())\n"
            "missing\n\n"
            "print('duplicate_rows:', duplicates)\n"
        )
    elif step.analysis_type == "bivariate":
        body = (
            f"cols = {cols_expr}\n"
            "use = [c for c in cols if c in df.columns]\n"
            "if not use:\n"
            "    df.head(10)\n"
            "else:\n"
            "    numeric = df[use].select_dtypes(include='number')\n"
            "    corr = numeric.corr(numeric_only=True)\n"
            "    corr\n"
        )
    elif step.analysis_type == "feature_specific":
        body = (
            f"cols = {cols_expr}\n"
            "use = [c for c in cols if c in df.columns]\n"
            "out = {}\n"
            "for c in use:\n"
            "    s = df[c]\n"
            "    s2 = pd.to_datetime(s, errors='coerce')\n"
            "    out[c] = {'n_parsed': int(s2.notna().sum()), "
            "'min': str(s2.min()), 'max': str(s2.max())}\n"
            "out\n"
        )
    elif step.analysis_type == "univariate":
        body = (
            f"cols = {cols_expr}\n"
            "use = [c for c in cols if c in df.columns]\n"
            "summary = {}\n"
            "for c in use:\n"
            "    s = df[c]\n"
            "    if getattr(s.dtype, 'kind', '') in {'i','u','f'}:\n"
            "        summary[c] = s.describe().to_dict()\n"
            "    else:\n"
            "        summary[c] = s.astype('string').value_counts(dropna=False).head(15)"
            ".to_dict()\n"
            "summary if summary else df[use].head(10)\n"
        )
    else:
        body = (
            f"cols = {cols_expr}\n"
            "use = [c for c in cols if c in df.columns]\n"
            "df[use].head(10) if use else df.head(10)\n"
        )

    retry_messages = list(state.get("retry_messages", []))
    if retry_messages:
        last = str(retry_messages[-1])
        last_block = last.replace("\n", "\n# ")
        retry_banner = f"\n# Previous error:\n# {last_block}\n\n"
    else:
        retry_banner = "\n"

    return {"generated_code": prelude + retry_banner + body}
