# 09 — Observability

## Observability Philosophy

LLM-based systems are fundamentally non-deterministic: the same input can produce different outputs. This makes observability even more critical than in traditional systems: without detailed traces, it is impossible to understand why an analysis produced a given result, why a step required 3 retries, or why the Critic negatively evaluated a section.

The system adopts a three-layer approach: LangSmith for LLM traces, structured logging for non-LLM system events, and callback handlers for alerting.

---

## LangSmith

LangSmith is the native observability platform for LangChain/LangGraph. It is enabled by setting the environment variables `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. From that point on, every graph run is automatically traced without code changes.

### What is traced automatically

LangSmith automatically captures for each graph run:

**At the session level**: initial input, final output, total duration, estimated token cost, used LLM provider and model, final LangGraph state.

**At the node level**: for each graph node and each subgraph—received input, produced output, duration, errors, and token usage for LLM nodes.

**At the LLM-call level**: for each model invocation—the full prompt (system + user + history), full response, input/output tokens, latency, and any function calls.

**At the tool level**: for each tool invocation—tool name, input, output, duration, and any errors.

### Custom metadata

LangGraph allows adding custom metadata to each run via config. The system adds:

```python
config = {
    "configurable": {"thread_id": session_id},
    "metadata": {
        "session_id": session_id,
        "dataset_name": file_name,
        "llm_provider": provider,
        "llm_model": model,
        "hitl_enabled": hitl_enabled,
        "mode": mode
    },
    "tags": ["eda-agent", f"provider:{provider}", f"mode:{mode}"]
}
```

These metadata allow filtering and grouping runs in LangSmith for aggregate analysis (e.g., "what is the average retry rate per provider?").

### Debugging use cases with LangSmith

**Why did this step require 3 retries?** By opening the trace, you can see the exact code generated on each attempt, the produced error, and the CODER response to the OBSERVER hint.

**Why did the Planner generate such a different plan?** You can inspect the full prompt sent to the Planner (including the serialized `DatasetContext`) and the structured response.

**How much does a full EDA cost on average?** LangSmith aggregates tokens per session. With custom metadata you can filter by provider/model.

**Which part of the pipeline produces more errors?** With tags you can group by step type and inspect error distributions.

---

## Structured Logging

In addition to LangSmith (which traces LLM calls), the system uses structured JSON logging for non-LLM system events.

### Domain-specific loggers

The system uses dedicated Python loggers per layer:

- `eda_agent.ingestion`: file loading and parsing events
- `eda_agent.executor`: code execution events in the kernel
- `eda_agent.graph`: LangGraph events (nodes, transitions, errors)
- `eda_agent.api`: HTTP request events
- `eda_agent.checkpointer`: checkpoint save/load events

### Log format

Each log entry is a JSON object with standardized fields:

```json
{
  "timestamp": "2026-01-15T10:23:45.123Z",
  "level": "INFO",
  "logger": "eda_agent.executor",
  "message": "Code executed successfully",
  "session_id": "session-uuid",
  "step_id": "dist_age",
  "execution_time_ms": 234,
  "output_types": ["text/plain", "image/png"],
  "code_lines": 12
}
```

JSON format enables direct ingestion into log management systems (Elasticsearch, Loki, Datadog, etc.) without custom parsing.

### Log levels and what to log

**DEBUG**: everything—full prompts, raw kernel outputs, LangGraph state after each node. Development only; too verbose for production.

**INFO**: meaningful business events—session started, step completed, section completed, analysis completed, session resumed from checkpoint.

**WARNING**: non-critical anomalies—step retry, skipped step, context window at 70%, LLM latency above threshold.

**ERROR**: handled errors—step permanently failed, LLM API error with retries in progress, kernel restarted.

**CRITICAL**: unhandled failures—main-process crash, checkpointer state corruption, unrecoverable database error.

---

## Callback Handlers for Alerting

LangChain provides the `BaseCallbackHandler` mechanism to intercept system events (LLM calls, tool calls, chain calls, errors) and react with custom logic. This system uses callback handlers for alerting on critical errors.

### `EDAErrorCallbackHandler`

This custom handler extends `BaseCallbackHandler` and overrides `on_chain_error`, `on_tool_error`, and `on_llm_error`.

When invoked, it evaluates error severity (critical vs non-critical) and, for critical errors, sends an alert. The delivery channel is configurable: email (SMTP), Slack webhook, generic webhook, or file logging (default in development).

The alert payload includes: session_id, the step where the error occurred, error type, full message, stack trace, and a direct link to the LangSmith trace (if available).

### `EDAMetricsCallbackHandler`

This handler collects aggregate metrics per session:

- Total tokens used (input + output, per node)
- Latency for each LLM invocation
- Retry count per step
- Estimated dollar cost (computed from tokens and provider pricing)

Metrics are written to a per-session JSON file and exposed via `GET /sessions/{id}/metrics`.

### Callback registration

Callback handlers are registered when a session is created and passed to LangGraph via the config `callbacks` parameter. This way they are automatically propagated to all LLM calls inside the graph, without explicitly passing them to each node.

---

## System Metrics

The system exposes operational metrics via `GET /metrics` in Prometheus-compatible format:

**`eda_sessions_total`**: total sessions count, labeled by `status` (completed, failed, in_progress).

**`eda_session_duration_seconds`**: histogram of completed session durations.

**`eda_steps_total`**: total steps executed, labeled by `result` (success, retry_success, failed).

**`eda_retry_count`**: distribution of retry counts per step.

**`eda_llm_tokens_total`**: total tokens used, labeled by `provider`, `model`, `agent` (planner, coder, observer, critic, conversation).

**`eda_llm_latency_seconds`**: histogram of LLM call latencies.

These metrics help monitor system health, identify cost trends, and optimize performance over time.
