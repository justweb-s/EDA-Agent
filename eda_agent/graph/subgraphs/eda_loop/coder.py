"""CODER node."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig

from eda_agent.config import EDAConfig
from eda_agent.graph.coercion import coerce_dataset_context, coerce_execution_history, coerce_step
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.llm_factory import get_llm
from eda_agent.models.execution import CodeCell
from eda_agent.models.session import SessionMetadata


def coder_node(state: EDALoopState, config: RunnableConfig | None = None) -> Any:

    step = coerce_step(state.get("current_step"))
    dataset_context = coerce_dataset_context(state.get("dataset_context"))
    if step is None or dataset_context is None:
        return {"generated_code": "raise ValueError('Missing current_step or dataset_context')"}

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

    retry_messages = list(state.get("retry_messages", []))
    if retry_messages:
        last = str(retry_messages[-1])
        last_block = last.replace("\n", "\n# ")
        retry_banner = f"\n# Previous error:\n# {last_block}\n\n"
    else:
        retry_banner = "\n"

    execution_history_json = json.dumps(
        [
            h.model_dump(mode="json")
            for h in coerce_execution_history(state.get("execution_history", []))
        ]
    )

    prompt = (
        "TASK: CODER\n"
        "Return a JSON object matching the CodeCell schema "
        "(keys: code, expected_output_description).\n"
        "Write minimal Python code for the current step.\n\n"
        f"STEP_JSON: {json.dumps(step.model_dump(mode='json'))}\n\n"
        f"DATASET_CONTEXT_JSON: {json.dumps(dataset_context.to_prompt_dict())}\n\n"
        f"EXECUTION_HISTORY_JSON: {execution_history_json}\n\n"
        f"RETRY_MESSAGES_JSON: {json.dumps(retry_messages)}"
    )

    try:
        if cfg.llm_provider.lower() == "mock":
            resp = llm.invoke([HumanMessage(content=prompt)])
            text = str(getattr(resp, "content", resp))
            cell = CodeCell.model_validate(json.loads(text))
        else:
            structured = llm.with_structured_output(CodeCell)
            out = structured.invoke([HumanMessage(content=prompt)])
            cell = out if isinstance(out, CodeCell) else CodeCell.model_validate(out)
    except Exception:
        resp = llm.invoke([HumanMessage(content=prompt)])
        text = str(getattr(resp, "content", resp))
        try:
            cell = CodeCell.model_validate(json.loads(text))
        except Exception:
            cell = CodeCell(
                code="raise RuntimeError('Coder LLM output parsing failed')",
                expected_output_description="",
            )

    return {
        "generated_code": prelude + retry_banner + str(cell.code).rstrip() + "\n",
        "expected_output_description": str(cell.expected_output_description),
    }
