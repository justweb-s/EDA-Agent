from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from fastapi import FastAPI

from eda_agent.ingestion.loader import load_file
from eda_agent.models.notebook import CellOutput, NotebookCell

from .session_store import SessionStore
from .sse_broker import SSEBroker


async def run_minimal_session(*, app: FastAPI, session_id: str) -> None:
    store = cast(SessionStore, app.state.session_store)
    broker = cast(SSEBroker, app.state.sse_broker)
    channel = await broker.get_channel(session_id)

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

        df = load_file(rec.file_path)

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

        describe = df.describe(include="all").transpose().head(25)
        describe_md = describe.to_markdown()

        code = (
            "import pandas as pd\n\n"
            f"# Source file on server: {rec.file_path}\n"
            "# Tip: place the dataset next to the notebook and adjust the path.\n\n"
            f'df = pd.read_csv(r"{rec.file_path}")\n'
            "df.describe(include='all').T\n"
        )

        cells.append(
            NotebookCell(
                cell_type="code",
                source=code,
                outputs=[
                    CellOutput(output_type="display_data", data={"text/markdown": describe_md})
                ],
                execution_count=1,
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
        await channel.close()
        await asyncio.sleep(0)
