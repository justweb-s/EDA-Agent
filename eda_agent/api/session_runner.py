from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from langgraph.types import Command

from eda_agent.api.event_translator import EventTranslator
from eda_agent.checkpointing.setup import build_checkpointer, checkpointer_metadata
from eda_agent.config import EDAConfig
from eda_agent.graph.parent.supervisor import build_supervisor_graph
from eda_agent.models.notebook import NotebookCell
from eda_agent.models.plan import EDAStep
from eda_agent.models.session import SessionMetadata
from eda_agent.tools.kernel import create_kernel

from .session_store import SessionStore
from .sse_broker import SSEBroker


async def run_minimal_session(*, app: FastAPI, session_id: str, resume: dict | None = None) -> None:
    config = cast(EDAConfig, app.state.config)
    store = cast(SessionStore, app.state.session_store)
    broker = cast(SSEBroker, app.state.sse_broker)
    channel = await broker.get_channel(session_id)

    kernel = None
    checkpointer = None
    should_close_channel = True

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

        cells: list[NotebookCell] = list(rec.notebook_cells or [])

        if not cells:
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

        checkpointer = build_checkpointer(config)
        supervisor = build_supervisor_graph(checkpointer=checkpointer)
        translator = EventTranslator()

        hitl_enabled = config.hitl_default_mode != "none"

        session_metadata = SessionMetadata(
            session_id=session_id,
            started_at=rec.created_at,
            llm_provider=config.llm_provider,
            llm_model=config.llm_model,
            file_name=rec.file_name,
        )

        graph_input: dict | Command = {
            "dataset_context": rec.dataset_context,
            "messages": [],
            "hitl_enabled": hitl_enabled,
            "session_metadata": session_metadata,
        }
        runnable_config = {
            "configurable": {"thread_id": session_id},
            "metadata": checkpointer_metadata(config, session_id, rec.file_name),
        }

        if resume is not None:
            graph_input = Command(resume=resume)

        loop = asyncio.get_running_loop()

        plan_event_emitted = False
        interrupt_event_emitted = False

        def _run_graph_stream() -> None:
            nonlocal plan_event_emitted, interrupt_event_emitted
            for ev in supervisor.stream(
                graph_input,
                runnable_config,
                stream_mode="updates",
                subgraphs=True,
            ):
                for sse_event, data in translator.translate(ev):
                    if sse_event == "plan_generated":
                        plan_event_emitted = True
                    if sse_event == "hitl_interrupt":
                        interrupt_event_emitted = True
                    fut = asyncio.run_coroutine_threadsafe(
                        channel.publish(sse_event, {"session_id": session_id, **data}),
                        loop,
                    )
                    fut.result(timeout=30)

        await asyncio.to_thread(_run_graph_stream)

        snapshot = await asyncio.to_thread(supervisor.get_state, runnable_config)
        values = cast(dict, getattr(snapshot, "values", {}))

        pending_interrupt: dict | None = None
        for task in getattr(snapshot, "tasks", ()) or ():
            interrupts = getattr(task, "interrupts", ()) or ()
            for it in interrupts:
                pending_interrupt = {
                    "value": getattr(it, "value", it),
                    "resumable": getattr(it, "resumable", None),
                    "ns": getattr(it, "ns", None),
                    "when": getattr(it, "when", None),
                }
                break
            if pending_interrupt is not None:
                break

        if resume is None and (interrupt_event_emitted or pending_interrupt is not None):
            if pending_interrupt is not None and not interrupt_event_emitted:
                await channel.publish(
                    "hitl_interrupt",
                    {"session_id": session_id, "interrupt": pending_interrupt},
                )
            suspended = replace(
                rec,
                status="suspended",
                n_cells=len(cells),
                notebook_cells=cells,
                error=None,
            )
            store.upsert(suspended)
            await channel.publish("session_suspended", {"session_id": session_id})
            should_close_channel = False
            return

        eda_plan_raw = list(values.get("eda_plan", []))
        eda_plan: list[EDAStep] = [
            (step if isinstance(step, EDAStep) else EDAStep.model_validate(step))
            for step in eda_plan_raw
        ]

        if eda_plan and not plan_event_emitted:
            await channel.publish(
                "plan_generated",
                {
                    "session_id": session_id,
                    "eda_plan": [s.model_dump(mode="json") for s in eda_plan],
                    "n_steps": len(eda_plan),
                },
            )

        if eda_plan:
            plan_md = "## Plan\n\n"
            for i, step in enumerate(eda_plan, start=1):
                cols = (
                    ", ".join(f"`{c}`" for c in step.target_columns) if step.target_columns else ""
                )
                cols_line = f"\\n  - **Columns**: {cols}" if cols else ""
                plan_md += (
                    f"{i}. **{step.section}** — {step.title}\\n  - {step.description}{cols_line}\\n"
                )

            cells.append(
                NotebookCell(
                    cell_type="markdown",
                    source=plan_md,
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

        kernel = create_kernel()
        await asyncio.to_thread(kernel.start, timeout_s=30)

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

        for step in eda_plan:
            step_md = f"## {step.section}: {step.title}\n\n{step.description}\n"
            if step.target_columns:
                step_md += (
                    "\n**Columns**: " + ", ".join(f"`{c}`" for c in step.target_columns) + "\n"
                )

            cells.append(
                NotebookCell(
                    cell_type="markdown",
                    source=step_md,
                    generated_at=datetime.now(UTC),
                    re_executable=True,
                    step_id=step.step_id,
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

            step_cols = list(step.target_columns or [])
            cols_expr = repr(step_cols)
            if step.analysis_type == "data_quality":
                step_code = (
                    "missing = df.isna().mean().sort_values(ascending=False).head(15)\n"
                    "duplicates = int(df.duplicated().sum())\n"
                    "missing\n\n"
                    "print('duplicate_rows:', duplicates)\n"
                )
            elif step.analysis_type == "bivariate":
                step_code = (
                    f"cols = {cols_expr}\n"
                    "use = [c for c in cols if c in df.columns]\n"
                    "numeric = df[use].select_dtypes(include='number')\n"
                    "corr = numeric.corr(numeric_only=True)\n"
                    "corr\n"
                )
            elif step.analysis_type == "feature_specific":
                step_code = (
                    f"cols = {cols_expr}\n"
                    "use = [c for c in cols if c in df.columns]\n"
                    "out = {}\n"
                    "for c in use:\n"
                    "    s = df[c]\n"
                    "    if s.dtype == 'object':\n"
                    "        s2 = pd.to_datetime(s, errors='coerce')\n"
                    "    else:\n"
                    "        s2 = pd.to_datetime(s, errors='coerce')\n"
                    "    out[c] = {'n_parsed': int(s2.notna().sum()), "
                    "'min': str(s2.min()), 'max': str(s2.max())}\n"
                    "out\n"
                )
            elif step.analysis_type == "univariate":
                step_code = (
                    f"cols = {cols_expr}\n"
                    "use = [c for c in cols if c in df.columns]\n"
                    "summary = {}\n"
                    "for c in use:\n"
                    "    s = df[c]\n"
                    "    if s.dtype.kind in {'i','u','f'}:\n"
                    "        summary[c] = s.describe().to_dict()\n"
                    "    else:\n"
                    "        summary[c] = s.astype('string').value_counts(dropna=False).head(15)"
                    ".to_dict()\n"
                    "summary\n"
                )
            else:
                step_code = (
                    f"cols = {cols_expr}\n"
                    "use = [c for c in cols if c in df.columns]\n"
                    "df[use].head(10)\n"
                )

            step_code = f"import pandas as pd\n\n{step_code}"
            step_result = await asyncio.to_thread(
                kernel.execute,
                step_code,
                timeout_s=config.kernel_execution_timeout,
                max_output_mb=config.kernel_max_output_size_mb,
            )

            cells.append(
                NotebookCell(
                    cell_type="code",
                    source=step_code,
                    outputs=step_result.cell_outputs,
                    execution_count=step_result.execution_count,
                    step_id=step.step_id,
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

        assembler_config = await asyncio.to_thread(
            supervisor.update_state,
            runnable_config,
            {"notebook_cells": cells},
            "plan_approval",
        )
        assembled = await asyncio.to_thread(supervisor.invoke, None, assembler_config)
        notebook_path = str(assembled.get("final_notebook_path") or "")
        if notebook_path:
            await channel.publish(
                "analysis_completed",
                {"session_id": session_id, "notebook_path": notebook_path},
            )

        completed = replace(
            rec,
            status="completed",
            n_cells=len(cells),
            notebook_cells=cells,
            notebook_path=notebook_path or None,
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
        if checkpointer is not None:
            try:
                conn = getattr(checkpointer, "conn", None)
                if conn is not None:
                    await asyncio.to_thread(conn.close)
            except Exception:
                pass
        if should_close_channel:
            await channel.close()
        await asyncio.sleep(0)
