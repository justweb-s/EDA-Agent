"""Checkpointer factory.

For single-user deployments, SQLite is the default persistence backend.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver

from eda_agent.config import EDAConfig


def build_checkpointer(config: EDAConfig) -> BaseCheckpointSaver:
    """Build the configured checkpointer."""

    backend = config.checkpoint_backend

    if backend == "memory":
        return MemorySaver()

    if backend == "sqlite":
        from langgraph.checkpoint.sqlite import SqliteSaver

        conn = sqlite3.connect(str(config.sqlite_path), check_same_thread=False)
        return SqliteSaver(conn)

    raise ValueError(f"Unsupported checkpoint backend: {backend}")


def checkpointer_metadata(config: EDAConfig, session_id: str, file_name: str) -> dict[str, Any]:
    """Return standard LangSmith/LangGraph metadata for a run config."""

    return {
        "session_id": session_id,
        "dataset_name": file_name,
        "llm_provider": config.llm_provider,
        "llm_model": config.llm_model,
    }
