"""Planner nodes."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from eda_agent.config import EDAConfig
from eda_agent.graph.coercion import coerce_dataset_context, coerce_plan
from eda_agent.graph.subgraphs.planner.state import PlannerState
from eda_agent.llm_factory import get_llm
from eda_agent.models.plan import EDAplan, EDAStep
from eda_agent.models.session import SessionMetadata


def _config_from_state(state: PlannerState) -> EDAConfig:
    cfg = EDAConfig()
    meta = state.get("session_metadata")
    if isinstance(meta, SessionMetadata):
        return cfg.model_copy(
            update={
                "llm_provider": meta.llm_provider,
                "llm_model": meta.llm_model,
            }
        )
    if isinstance(meta, dict):
        meta2 = SessionMetadata.model_validate(meta)
        return cfg.model_copy(
            update={"llm_provider": meta2.llm_provider, "llm_model": meta2.llm_model}
        )
    return cfg


def _get_llm(state: PlannerState, config: RunnableConfig | None) -> Any:
    cfg = _config_from_state(state)
    callbacks = None
    if config is not None:
        callbacks = cast(Any, config.get("callbacks"))
    return get_llm(cfg, callbacks=callbacks)


def _dataset_column_names(dataset_context: Any) -> set[str]:
    if dataset_context is None:
        return set()
    cols = getattr(dataset_context, "columns", None) or []
    out: set[str] = set()
    for c in cols:
        name = getattr(c, "name", None)
        if name:
            out.add(str(name))
    return out


def _validate_dag(steps: list[EDAStep]) -> list[str]:
    errors: list[str] = []
    ids = [s.step_id for s in steps]
    id_set = set(ids)

    if len(id_set) != len(ids):
        dupes = sorted({x for x in ids if ids.count(x) > 1})
        errors.append(f"Duplicate step_id values: {dupes}")

    for s in steps:
        for dep in s.depends_on:
            if dep not in id_set:
                errors.append(f"Step '{s.step_id}' depends on unknown step_id '{dep}'")

    graph: dict[str, list[str]] = {s.step_id: list(s.depends_on) for s in steps}
    visiting: set[str] = set()
    visited: set[str] = set()

    def dfs(node: str) -> bool:
        if node in visiting:
            return True
        if node in visited:
            return False
        visiting.add(node)
        for nxt in graph.get(node, []):
            if nxt in graph and dfs(nxt):
                return True
        visiting.remove(node)
        visited.add(node)
        return False

    for nid in graph:
        if dfs(nid):
            errors.append("Dependency cycle detected in plan")
            break

    return errors


def analyze_dataset(*args: Any, **kwargs: Any) -> Any:
    state = cast(PlannerState, args[0] if args else (kwargs.get("state") or {}))
    config = cast(
        RunnableConfig | None,
        (args[1] if len(args) > 1 else kwargs.get("config")),
    )

    dataset_context = coerce_dataset_context(state.get("dataset_context"))
    if dataset_context is None:
        return {
            "dataset_analysis": "",
            "validation_errors": ["Missing dataset_context"],
        }

    llm = _get_llm(state, config)
    base_messages = list(state.get("messages", []))

    prompt = (
        "TASK: ANALYZE_DATASET\n"
        "You are an EDA planning assistant. Summarize the dataset context and "
        "highlight key risks/next steps.\n\n"
        f"DATASET_CONTEXT_JSON: {json.dumps(dataset_context.to_prompt_dict())}"
    )
    resp = llm.invoke(base_messages + [HumanMessage(content=prompt)])
    text = str(getattr(resp, "content", resp))
    base_messages.append(AIMessage(content=text))

    return {
        "messages": base_messages,
        "dataset_analysis": text,
        "validation_errors": list(state.get("validation_errors", [])),
    }


def generate_plan(*args: Any, **kwargs: Any) -> Any:
    state = cast(PlannerState, args[0] if args else (kwargs.get("state") or {}))
    config = cast(
        RunnableConfig | None,
        (args[1] if len(args) > 1 else kwargs.get("config")),
    )

    dataset_context = coerce_dataset_context(state.get("dataset_context"))
    if dataset_context is None:
        return {
            "draft_plan": [],
            "validation_errors": ["Missing dataset_context"],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }

    llm = _get_llm(state, config)
    messages = list(state.get("messages", []))
    analysis = str(state.get("dataset_analysis") or "")
    validation_errors = list(state.get("validation_errors", []))

    prompt = (
        "TASK: GENERATE_PLAN\n"
        "Return a JSON object matching the EDAplan schema (key: steps).\n"
        "Each step must have a unique step_id, valid depends_on references, and "
        "target_columns must exist.\n\n"
        f"DATASET_CONTEXT_JSON: {json.dumps(dataset_context.to_prompt_dict())}\n\n"
        f"DATASET_ANALYSIS: {analysis}\n\n"
        f"VALIDATION_ERRORS: {json.dumps(validation_errors)}"
    )

    try:
        if _config_from_state(state).llm_provider.lower() == "mock":
            resp = llm.invoke(messages + [HumanMessage(content=prompt)])
            text = str(getattr(resp, "content", resp))
            plan = EDAplan.model_validate(json.loads(text))
            return {
                "draft_plan": plan.steps,
                "iteration_count": int(state.get("iteration_count", 0)) + 1,
            }
        structured = llm.with_structured_output(EDAplan)
        out = structured.invoke(messages + [HumanMessage(content=prompt)])
        plan = out if isinstance(out, EDAplan) else EDAplan.model_validate(out)
        steps = plan.steps
        return {
            "draft_plan": steps,
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }
    except Exception as e:  # noqa: BLE001
        try:
            resp = llm.invoke(messages + [HumanMessage(content=prompt)])
            text = str(getattr(resp, "content", resp))
            plan = EDAplan.model_validate(json.loads(text))
            return {
                "draft_plan": plan.steps,
                "iteration_count": int(state.get("iteration_count", 0)) + 1,
            }
        except Exception:
            pass
        return {
            "draft_plan": [],
            "validation_errors": [f"LLM plan parsing error: {e}"],
            "iteration_count": int(state.get("iteration_count", 0)) + 1,
        }


def validate_plan(*args: Any, **kwargs: Any) -> Any:
    state = cast(PlannerState, args[0] if args else (kwargs.get("state") or {}))

    dataset_context = coerce_dataset_context(state.get("dataset_context"))
    steps = coerce_plan(state.get("draft_plan", []))

    errors: list[str] = []
    if dataset_context is None:
        errors.append("Missing dataset_context")
    if not steps:
        errors.append("Planner produced an empty plan")

    col_names = _dataset_column_names(dataset_context)
    for s in steps:
        if s.target_columns:
            missing = [c for c in s.target_columns if c not in col_names]
            if missing:
                errors.append(f"Step '{s.step_id}' refers to missing columns: {missing}")

    errors.extend(_validate_dag(steps))

    if errors:
        return {"validation_errors": errors}

    return {
        "eda_plan": steps,
        "validation_errors": [],
    }
