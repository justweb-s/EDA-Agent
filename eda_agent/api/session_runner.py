from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

from fastapi import FastAPI
from langchain_core.messages import HumanMessage
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
        store.upsert(replace(rec, status="in_progress", error=None))
        await channel.publish(
            "session_started",
            {
                "session_id": session_id,
                "dataset_name": rec.file_name,
                "llm_provider": rec.llm_provider,
                "llm_model": rec.llm_model,
                "mode": rec.mode,
                "hitl_enabled": rec.hitl_enabled,
            },
        )

        cells: list[NotebookCell] = [
            (cell if isinstance(cell, NotebookCell) else NotebookCell.model_validate(cell))
            for cell in list(rec.notebook_cells or [])
        ]

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
                    "cell_type": "markdown",
                    "content": overview_md,
                    "outputs": [],
                    "step_id": None,
                    "execution_count": None,
                    "generated_at": cells[-1].generated_at.isoformat(),
                    "re_executable": True,
                },
            )

        checkpointer = build_checkpointer(config)
        supervisor = build_supervisor_graph(checkpointer=checkpointer)
        translator = EventTranslator()

        hitl_enabled = bool(rec.hitl_enabled)

        translator._last_n_cells = len(cells)

        session_metadata = SessionMetadata(
            session_id=session_id,
            started_at=rec.created_at,
            llm_provider=rec.llm_provider,
            llm_model=rec.llm_model,
            file_name=rec.file_name,
        )

        messages = []
        if rec.user_instructions:
            messages = [HumanMessage(content=str(rec.user_instructions))]

        graph_input: dict | Command = {
            "dataset_context": rec.dataset_context,
            "messages": messages,
            "mode": rec.mode,
            "hitl_enabled": hitl_enabled,
            "session_metadata": session_metadata,
            "notebook_cells": cells,
            "current_step_index": 0,
            "execution_history": [],
        }
        runnable_config = {
            "configurable": {"thread_id": session_id},
            "metadata": checkpointer_metadata(
                config,
                session_id,
                rec.file_name,
                llm_provider=rec.llm_provider,
                llm_model=rec.llm_model,
            ),
        }

        if resume is not None:
            graph_input = Command(resume=resume)

            snapshot0 = await asyncio.to_thread(supervisor.get_state, runnable_config)
            values0 = cast(dict, getattr(snapshot0, "values", {}))
            translator._last_n_cells = len(list(values0.get("notebook_cells", [])))

        needs_kernel = (resume is not None) or (not hitl_enabled)
        if needs_kernel:
            kernel = create_kernel()
            await asyncio.to_thread(kernel.start, timeout_s=30)
            runnable_config["configurable"]["kernel"] = kernel

        loop = asyncio.get_running_loop()

        plan_event_emitted = resume is not None
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
                        if plan_event_emitted:
                            continue
                        plan_event_emitted = True
                    if sse_event == "hitl_interrupt":
                        if interrupt_event_emitted:
                            continue
                        interrupt_event_emitted = True
                    fut = asyncio.run_coroutine_threadsafe(
                        channel.publish(sse_event, {"session_id": session_id, **data}),
                        loop,
                    )
                    fut.result(timeout=30)

        await asyncio.to_thread(_run_graph_stream)

        snapshot = await asyncio.to_thread(supervisor.get_state, runnable_config)
        values = cast(dict, getattr(snapshot, "values", {}))

        notebook_cells_raw = list(values.get("notebook_cells", []))
        cells_from_graph: list[NotebookCell] = [
            (cell if isinstance(cell, NotebookCell) else NotebookCell.model_validate(cell))
            for cell in notebook_cells_raw
        ]

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
                interrupt_value = pending_interrupt.get("value")
                interrupt_type = None
                if isinstance(interrupt_value, dict):
                    interrupt_type = interrupt_value.get("type")

                available_actions: list[str] = []
                if interrupt_type == "plan_approval":
                    available_actions = ["approve", "reject", "modify"]

                await channel.publish(
                    "hitl_interrupt",
                    {
                        "session_id": session_id,
                        "interrupt": pending_interrupt,
                        "interrupt_type": interrupt_type,
                        "data": interrupt_value,
                        "available_actions": available_actions,
                    },
                )
            suspended = replace(
                rec,
                status="suspended",
                n_cells=len(cells_from_graph),
                notebook_cells=cells_from_graph,
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

        if resume is None and eda_plan and not plan_event_emitted:
            await channel.publish(
                "plan_generated",
                {
                    "session_id": session_id,
                    "eda_plan": [s.model_dump(mode="json") for s in eda_plan],
                    "n_steps": len(eda_plan),
                },
            )

        cells = cells_from_graph

        notebook_path = str(values.get("final_notebook_path") or "")

        completed = replace(
            rec,
            status="completed",
            n_cells=len(cells),
            notebook_cells=cells,
            notebook_path=notebook_path or None,
            error=None,
        )
        store.upsert(completed)

        if notebook_path:
            await channel.publish(
                "analysis_completed",
                {
                    "session_id": session_id,
                    "notebook_path": notebook_path,
                    "n_cells": len(cells),
                },
            )

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
        await channel.publish(
            "error",
            {
                "session_id": session_id,
                "error": {
                    "code": "SESSION_FAILED",
                    "message": str(e),
                    "details": {"session_id": session_id},
                    "recoverable": False,
                    "suggested_action": "Inspect server logs",
                },
            },
        )
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
