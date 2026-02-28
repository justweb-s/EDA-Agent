from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI

from eda_agent.config import EDAConfig
from eda_agent.models.notebook import NotebookCell
from eda_agent.tools.kernel import create_kernel

from .session_store import SessionStore
from .sse_broker import SSEBroker


async def run_minimal_session(*, app: FastAPI, session_id: str) -> None:
    config = cast(EDAConfig, app.state.config)
    store = cast(SessionStore, app.state.session_store)
    broker = cast(SSEBroker, app.state.sse_broker)
    channel = await broker.get_channel(session_id)

    kernel = None

    rec = store.get(session_id)
    if rec is None:
        await channel.publish(
            "session_failed", {"session_id": session_id, "error": "Session not found"}
        )
        await channel.close()
        return

    try:
        store.upsert(replace(rec, status="running", error=None))
        await channel.publish("session_started", {"session_id": session_id})

        kernel = create_kernel()
        await asyncio.to_thread(kernel.start, timeout_s=30)

        cells: list[NotebookCell] = []

        overview_md = (
            f"# EDA Agent\n\n"
            f"**File**: `{rec.file_name}`\n\n"
            f"**Shape**: `{rec.dataset_context.shape[0]}` rows x "
            f"`{rec.dataset_context.shape[1]}` columns\n\n"
            f"## Detected issues\n\n"
        )
        if rec.dataset_context.detected_issues:
            for issue in rec.dataset_context.detected_issues:
                col = f" (`{issue.column}`)" if issue.column else ""
                overview_md += (
                    f"- **{issue.severity.upper()}** `{issue.code}`{col}: {issue.message}\n"
                )
        else:
            overview_md += "- None\n"

        cells.append(
            NotebookCell(
                cell_type="markdown",
                source=overview_md,
                generated_at=datetime.now(UTC),
                re_executable=True,
            )
        )
        await channel.publish(
            "cell_added",
            {
                "session_id": session_id,
                "n_cells": len(cells),
                "cell": cells[-1].model_dump(mode="json"),
            },
        )

        suffix = Path(rec.file_path).suffix.lower()
        read_stmt = (
            f'df = pd.read_excel(r"{rec.file_path}")'
            if suffix in {".xls", ".xlsx"}
            else f'df = pd.read_csv(r"{rec.file_path}")'
        )

        code = (
            "import pandas as pd\n\n"
            f"# Source file on server: {rec.file_path}\n"
            "# Tip: place the dataset next to the notebook and adjust the path.\n\n"
            f"{read_stmt}\n\n"
            "describe = df.describe(include='all').T.head(25)\n"
            "describe\n"
        )

        run_result = await asyncio.to_thread(
            kernel.execute,
            code,
            timeout_s=config.kernel_execution_timeout,
            max_output_mb=config.kernel_max_output_size_mb,
        )

        cells.append(
            NotebookCell(
                cell_type="code",
                source=code,
                outputs=run_result.cell_outputs,
                execution_count=run_result.execution_count,
                step_id="overview",
                generated_at=datetime.now(UTC),
                re_executable=False,
            )
        )
        await channel.publish(
            "cell_added",
            {
                "session_id": session_id,
                "n_cells": len(cells),
                "cell": cells[-1].model_dump(mode="json"),
            },
        )

        completed = replace(
            rec,
            status="completed",
            n_cells=len(cells),
            notebook_cells=cells,
            error=None,
        )
        store.upsert(completed)

        await channel.publish(
            "session_completed",
            {"session_id": session_id, "n_cells": len(cells)},
        )

    except Exception as e:  # noqa: BLE001
        failed = replace(
            rec,
            status="failed",
            error=str(e),
        )
        store.upsert(failed)
        await channel.publish("session_failed", {"session_id": session_id, "error": str(e)})

    finally:
        if kernel is not None:
            try:
                await asyncio.to_thread(kernel.shutdown)
            except Exception:
                pass
        await channel.close()
        await asyncio.sleep(0)
