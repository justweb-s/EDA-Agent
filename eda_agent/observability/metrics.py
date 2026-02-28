"""Prometheus metrics definitions."""

from __future__ import annotations

from prometheus_client import Counter, Histogram

eda_sessions_total = Counter(
    "eda_sessions_total",
    "Total number of EDA sessions.",
    labelnames=("status",),
)

eda_session_duration_seconds = Histogram(
    "eda_session_duration_seconds",
    "Duration of completed EDA sessions in seconds.",
)

eda_steps_total = Counter(
    "eda_steps_total",
    "Total number of EDA steps executed.",
    labelnames=("result",),
)

eda_retry_count = Histogram(
    "eda_retry_count",
    "Distribution of retry counts per EDA step.",
)

eda_llm_tokens_total = Counter(
    "eda_llm_tokens_total",
    "Total LLM tokens used.",
    labelnames=("provider", "model", "agent"),
)

eda_llm_latency_seconds = Histogram(
    "eda_llm_latency_seconds",
    "LLM call latency in seconds.",
    labelnames=("provider", "model", "agent"),
)
