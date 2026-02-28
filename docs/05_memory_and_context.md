# 05 — Memory and Context

## Memory Types in the System

The system uses three types of "memory" with different purposes and mechanisms. It is important not to confuse them.

### 1. Working memory (short-term / in-session)

The `messages` list in state is the working memory of the current session. It contains the full conversation history: user messages, agent responses, tool results. It is managed via LangGraph's `MessagesAnnotation` and persisted by the checkpointer.

This memory lives for the full duration of a session (from file upload to final notebook generation) and does not survive session termination.

### 2. Semantic analysis memory (execution history)

`execution_history` is a specialized form of memory that does not contain LLM messages but **structured semantic findings** extracted from code execution. Every completed step appends an `ExecutionSummary` with a natural-language description of what was found.

This memory is the key mechanism behind adaptivity: the CODER reads it before generating new code to avoid repetition and to adapt the analysis to prior results.

### 3. Foundational context (`DatasetContext`)

`DatasetContext` is technically memory—it contains dataset information—but it is immutable and always present. It is never compressed or removed. It is the system's "ground truth".

---

## The Context Window Problem

A full EDA session on a real dataset can generate many messages: the Planner plan, messages from each CODER/OBSERVER cycle, Critic reviews, and the user conversation. The sum can easily exceed the chosen LLM's context window.

This is most visible with smaller context-window models (e.g., local models via Ollama), but it can also affect larger-context models if analysis is extensive.

Exceeding the context window typically does not produce a controlled error—it usually results in a provider API error that interrupts execution. This is particularly problematic for a system that runs long sessions.

---

## Context Window Management Strategy

The system uses a three-layer strategy.

### Layer 1: Separate LLM messages from findings

The most important point is the **strict separation** between `messages` and findings in `execution_history`. Messages exchanged between CODER and OBSERVER for each step are stored exclusively in the EDA Loop subgraph's internal state, not in the parent's `messages` list.

The parent's `messages` list contains only:

- The initial message with user instructions
- Conversation-agent messages with the user
- A summary for each completed section (not all intermediate messages)

This drastically reduces growth of `messages`.

### Layer 2: Progressive summarization

When `messages` exceeds a configurable threshold (default: 80% of the selected model's context window), a dedicated `SUMMARIZE_HISTORY` node is automatically invoked by the Supervisor.

This node uses LangChain's `SummarizationMiddleware` (or an equivalent implementation) to compress older messages into a structured summary. The summary is domain-specific and includes:

- Analysis goal
- Completed sections and their status
- Key findings per section (extracted from `execution_history`)
- Remaining plan steps
- Key decisions made during the session (e.g., skipped steps, plan modifications)

The most recent messages (configurable, default: last 10) are kept verbatim to preserve immediate context.

The summary is inserted as a `SystemMessage` at the beginning of the compressed message list, so every subsequent LLM call retains the full-session context.

### Layer 3: `DatasetContext` is always present

`DatasetContext` is never part of the compressed `messages` list—it is managed separately as a state field. It is injected into each LLM system prompt via a dedicated prompt-building mechanism, not as a conversational message.

This ensures that even after summarization, every agent always has access to foundational dataset information.

---

## `DatasetContext`: Construction and Contents

`DatasetContext` is built during Data Ingestion, before starting the LangGraph graph. This is pure Python code and does not involve an LLM.

### Automatic column analysis

For each DataFrame column, the system computes:

**Numeric columns**: count, null_count, null_percentage, min, max, mean, median, std, skewness, kurtosis, quartiles (Q1, Q3), IQR, n_unique, n_zeros. It also automatically detects potential outliers (values beyond 3σ) and the approximate distribution shape (normal, right/left-skewed, bimodal).

**Categorical columns**: count, null_count, null_percentage, n_unique, top_values (top 10 values with frequency), detected_encoding (ordinal vs nominal heuristics).

**Datetime columns**: count, null_count, null_percentage, min_date, max_date, range_days, detected_frequency (daily, monthly, etc.), n_unique_days.

**Free-text columns**: count, null_count, avg_length, max_length, n_unique, top_bigrams (to infer text domain).

### Semantic classification

Each column receives a `detected_semantic_type` based on heuristics over the column name and values:

- `id_column`: all values unique, often progressive integers or UUIDs
- `target_variable`: column name contains "target", "label", "y", "outcome"
- `datetime`: detected automatically by Pandas
- `boolean`: only 2 distinct values
- `categorical_low_cardinality`: categorical with fewer than 20 unique values
- `categorical_high_cardinality`: categorical with 20+ unique values
- `numeric_continuous`: numeric with many unique values
- `numeric_discrete`: integer numeric with few unique values
- `text_free`: long strings, high cardinality

This semantic classification is used by the Planner to decide which analyses to include and which to skip.

### Automatically detected issues

The system detects and documents in `DatasetContext`:

- Columns with null percentage above configurable thresholds (warning: 5%, error: 30%)
- Single-value columns (constants—often removable)
- Duplicate columns (same contents, different name)
- Potential primary keys (all values unique)
- Imbalanced datasets (if a target column is detected with a highly imbalanced distribution)
- Potentially incorrect dtypes (e.g., numeric stored as string)

These detected issues are included in the Planner input, so it can add targeted data-quality steps.

---

## Execution History: Adaptive Memory

`execution_history` is the most critical memory mechanism for analysis quality. Its design deserves special attention.

### What each `ExecutionSummary` contains

The `ExecutionSummary` produced by the Observer after each completed step is not a dump of raw output. It is a structured semantic summary:

```
ExecutionSummary {
  step_id: "dist_age"
  section: "distributions"
  findings: "Column `age` has an approximately normal distribution
             (skewness=0.23, kurtosis=2.8), mean 35.2±12.1 years.
             3 outliers detected (age > 75 years, <0.1%).
             No null values. Distribution is consistent with
             a general adult population."
  key_statistics: {mean: 35.2, std: 12.1, min: 18, max: 89, n_outliers: 3}
  charts_produced: ["histogram with KDE for age"]
  anomalies_found: []
  columns_analyzed: ["age"]
}
```

### How execution history is used

When the CODER generates code for step N, it receives as context:

1. Full findings from steps in the **current section** (high relevance, maximum detail)
2. A per-section summary of steps in **previous sections** (medium relevance, compressed format)
3. Anomalies found in any previous step (always included; they may affect future analyses)

This triage ensures the CODER always has relevant context without wasting tokens.

### Example of adaptivity driven by execution history

If the age distribution reveals a bimodal distribution with peaks at 25 and 55, the CODER generating the correlation analysis between age and income will automatically include segmentation for the two populations—not because it was explicitly instructed to, but because the prior step findings are in its context.

This is the desired "intelligent" behavior: not an agent that follows rigid instructions, but an agent that adapts analyses based on what it is discovering.
