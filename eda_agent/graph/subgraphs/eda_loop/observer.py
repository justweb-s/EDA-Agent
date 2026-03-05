"""OBSERVER node."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from eda_agent.config import EDAConfig
from eda_agent.graph.coercion import (
    coerce_execution_result,
    coerce_step,
)
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.llm_factory import get_llm
from eda_agent.models.execution import ObserverVerdict
from eda_agent.models.notebook import CellOutput
from eda_agent.models.session import SessionMetadata


def observer_node(state: EDALoopState, config: RunnableConfig | None = None) -> Any:

    step = coerce_step(state.get("current_step"))
    execution = coerce_execution_result(state.get("execution_result"))
    if step is None or execution is None:
        return {
            "observer_verdict": "fatal_error",
            "retry_messages": ["Missing current_step or execution_result"],
        }

    cfg = EDAConfig()
    meta = state.get("session_metadata")
    if isinstance(meta, SessionMetadata):
        cfg = cfg.model_copy(
            update={"llm_provider": meta.llm_provider, "llm_model": meta.llm_model}
        )
    elif isinstance(meta, dict):
        meta2 = SessionMetadata.model_validate(meta)
        cfg = cfg.model_copy(
            update={"llm_provider": meta2.llm_provider, "llm_model": meta2.llm_model}
        )

    callbacks = None
    if config is not None:
        callbacks = cast(Any, config.get("callbacks"))
    llm = get_llm(cfg, callbacks=callbacks)

    local_retry_count = int(state.get("local_retry_count", 0))
    retry_messages = list(state.get("retry_messages", []))

    expected = str(state.get("expected_output_description") or "")
    generated_code = str(state.get("generated_code") or "")
    outputs_raw = list(state.get("cell_outputs", []))
    outputs: list[dict[str, Any]] = []
    for o in outputs_raw:
        if isinstance(o, CellOutput):
            outputs.append(o.model_dump(mode="json"))
        elif isinstance(o, dict):
            outputs.append(o)
        else:
            outputs.append(
                {
                    "output_type": str(getattr(o, "output_type", "unknown")),
                    "text": str(o),
                }
            )

    prompt = (
        "TASK: OBSERVER\n"
        "Return a JSON object matching the ObserverVerdict schema.\n\n"
        f"STEP_JSON: {json.dumps(step.model_dump(mode='json'))}\n\n"
        f"EXPECTED_OUTPUT_DESCRIPTION: {expected}\n\n"
        f"CODE: {generated_code}\n\n"
        f"EXECUTION_SUCCESS: {json.dumps(bool(execution.success))}\n\n"
        f"STDOUT: {execution.stdout}\n\n"
        f"STDERR: {execution.stderr}\n\n"
        f"CELL_OUTPUTS_JSON: {json.dumps(outputs)}"
    )

    verdict_obj: ObserverVerdict | None = None
    try:
        if cfg.llm_provider.lower() == "mock":
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = str(getattr(resp, "content", resp))
            verdict_obj = ObserverVerdict.model_validate(json.loads(text))
        else:
            structured = llm.with_structured_output(ObserverVerdict)
            out = structured.invoke([HumanMessage(content=prompt)])
            verdict_obj = (
                out if isinstance(out, ObserverVerdict) else ObserverVerdict.model_validate(out)
            )
    except Exception:
        verdict_obj = None

    if not execution.success:
        error_msg = execution.stderr or "Execution failed"
        retry_messages.append(error_msg)
        if local_retry_count < (cfg.max_step_retries - 1):
            if verdict_obj is None:
                verdict_obj = ObserverVerdict(
                    verdict="retry",
                    findings_description=error_msg,
                    error_analysis=error_msg,
                    retry_hint="Fix the error and retry.",
                )
            if verdict_obj.verdict != "fatal_error":
                return {
                    "observer_verdict": "retry",
                    "observer_verdict_obj": verdict_obj.model_dump(mode="json"),
                    "retry_messages": retry_messages,
                    "local_retry_count": local_retry_count + 1,
                }

        return {
            "observer_verdict": "fatal_error",
            "local_retry_count": 0,
            "retry_messages": [],
            "observer_verdict_obj": (
                verdict_obj.model_dump(mode="json")
                if verdict_obj is not None
                else {"verdict": "fatal_error", "findings_description": error_msg}
            ),
        }

    final_verdict = "success"
    if verdict_obj is not None and verdict_obj.verdict in {"success", "fatal_error"}:
        final_verdict = verdict_obj.verdict

    return {
        "observer_verdict": final_verdict,
        "local_retry_count": 0,
        "retry_messages": [],
        "observer_verdict_obj": (
            verdict_obj.model_dump(mode="json")
            if verdict_obj is not None
            else {"verdict": final_verdict, "findings_description": "Step executed."}
        ),
    }
