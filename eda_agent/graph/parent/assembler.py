"""Notebook assembler node."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from eda_agent.api.notebook_export import export_ipynb_bytes
from eda_agent.config import EDAConfig
from eda_agent.graph.coercion import coerce_session_metadata
from eda_agent.graph.parent.state import EDAState
from eda_agent.models.notebook import NotebookCell


def assemble_notebook(*args: Any, **kwargs: Any) -> Any:
    state = cast(EDAState, args[0] if args else kwargs.get("state"))
    config = EDAConfig()

    session_metadata = coerce_session_metadata(state.get("session_metadata"))
    session_id = (
        getattr(session_metadata, "session_id", None) if session_metadata is not None else None
    )
    if not session_id:
        session_id = "unknown"

    cells_raw = state.get("notebook_cells") or []
    cells = [
        c if isinstance(c, NotebookCell) else NotebookCell.model_validate(c) for c in cells_raw
    ]

    notebooks_dir = (config.output_dir / "notebooks").resolve()
    notebooks_dir.mkdir(parents=True, exist_ok=True)
    notebook_path = (notebooks_dir / f"eda-agent-{session_id}.ipynb").resolve()
    tmp = Path(str(notebook_path) + ".tmp")
    tmp.write_bytes(export_ipynb_bytes(cells))
    tmp.replace(notebook_path)

    return {"final_notebook_path": str(notebook_path)}
