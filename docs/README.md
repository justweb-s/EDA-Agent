# EDA Agent — Project Documentation

> Multi-agent system built on LangGraph that automatically generates exploratory data analysis (EDA) as complete, runnable Jupyter Notebooks.

---

## Documentation Index

| # | File | Contents |
|---|------|----------|
| 01 | [Overall Architecture](./01_architecture.md) | Big-picture view, design philosophy, architectural patterns |
| 02 | [State Management](./02_state_management.md) | `EDAState`, subgraph states, annotations, reducers |
| 03 | [LangGraph Primitives](./03_langgraph_primitives.md) | Checkpointing, interrupts, conditional edges, subgraphs |
| 04 | [Agents and Subgraphs](./04_agents.md) | Planner, EDA Loop, Critic, Conversation — responsibilities, tools, prompting strategy |
| 05 | [Memory and Context](./05_memory_and_context.md) | Short-term memory, summarization, `DatasetContext`, context-window management |
| 06 | [Streaming](./06_streaming.md) | Streaming modes, SSE events, UI protocol |
| 07 | [Human-in-the-Loop](./07_human_in_the_loop.md) | Interrupts, `Command` resume, configurable touchpoints |
| 08 | [Robustness and Resilience](./08_robustness.md) | Retry strategy, crash resume, error handling, fallback |
| 09 | [Observability](./09_observability.md) | LangSmith, callback handlers, structured logging, alerting |
| 10 | [API and UI](./10_api_and_ui.md) | FastAPI endpoints, SSE, chat UI design |
| 11 | [Project Structure](./11_project_structure.md) | Directory layout, responsibilities of each module |
| 12 | [Configuration](./12_configuration.md) | `EDAConfig`, environment variables, dev/prod profiles |

---

## System Overview

### What it does

EDA Agent takes a CSV/Excel file as input and produces a complete, runnable Jupyter Notebook (`.ipynb`) that includes:

- Structured EDA that adapts to the specific dataset
- Executable Python code that is actually run during generation (cell outputs are captured)
- LLM-generated interpretation of key results
- Visualizations (rendered inline in the notebook)
- Adaptive sections driven by what is found during execution

### High-level flow

```
Input: CSV/Excel file + optional instructions
         │
         ▼
 [Data Ingestion]  →  Builds DatasetContext (schema, stats, sample)
         │
         ▼
 [PLANNER Agent]   →  Produces an adaptive, structured EDA plan
         │
         ▼
 [EDA LOOP Agent]  →  For each step: generate code → execute → observe output
         │
         ▼
 [CRITIC Agent]    →  Reviews quality of each section and requests improvements
         │
         ▼
 [ASSEMBLER]       →  Assembles all cells into a valid .ipynb
         │
         ▼
Output: complete .ipynb + optional chat session
```

### Core design principles

**Small steps, minimal code.** Each agent writes the minimum code necessary for a single objective, then observes the output before proceeding. The system avoids monolithic code blocks. This drastically reduces execution errors and makes retries more surgical.

**Data-driven adaptivity.** Every next decision is informed by what was actually discovered in prior steps. The plan is not rigid: it evolves based on executed outputs. If step 3 discovers a bimodal distribution, step 4 can be updated to investigate the two populations separately.

**LLM-agnostic.** The system works with any LangChain-supported LLM provider (OpenAI, Anthropic, Ollama, Groq, etc.) selected at runtime, without code changes.

**Automatic vs conversational dual-mode.** The same LangGraph graph supports a fully automatic single-shot mode and a chat mode where the user can ask questions and request changes during or after the analysis.

**Strict separation of responsibilities.** Each agent has one purpose, a prompt optimized for that purpose, and a minimal toolset. No agent does everything.

---

## Tech Stack

| Layer | Technology | Why |
|-------|------------|-----|
| Agent orchestration | LangGraph (StateGraph, subgraphs) | Explicit control flow, persistent state, native HITL primitives |
| LLM abstraction | LangChain (`BaseChatModel`) | Provider-agnostic, structured output, streaming |
| Checkpointing | LangGraph `SqliteSaver` (dev + prod) | No external dependencies, appropriate for single-user |
| Code execution | IPython kernel via `jupyter_client` | Persistent Python state across cells, isolation, timeouts |
| Notebook output | `nbformat` (official Jupyter lib) | Produces valid, re-runnable `.ipynb` files |
| Backend API | FastAPI (async) | SSE-friendly, async-first, integrates with LangGraph `astream` |
| UI streaming → browser | Server-Sent Events (SSE) | One-way server→client, simpler than WebSockets, auto-reconnect |
| Observability | LangSmith + custom `BaseCallbackHandler` | Full traces, run analytics, error alerting |
| Configuration | Pydantic `BaseSettings` | Type-safe env vars, multiple profiles |
| LLM output validation | Pydantic v2 + `with_structured_output` | Eliminates fragile parsing, enforces contracts |

---

## Operating Modes

### Single-shot (automatic)

The system runs the full analysis end-to-end without human intervention and produces the final notebook. Suitable for batch pipelines or users who want a complete output immediately.

### Interactive chat

The system pauses after each EDA section and waits for user input before continuing. The user can ask questions, request additional analyses, modify the plan, or approve and proceed.

### Single-shot with HITL

Same as single-shot, but the system pauses at key moments: after plan generation (the user can approve/modify) and after each Critic review (the user can accept or request deeper improvements). Suitable for high-stakes analyses where supervision is desired without giving up automation.

---

## System Requirements

- Python 3.11+
- Access to the chosen LLM provider APIs
- Environment variables configured (see [Configuration](./12_configuration.md))
- A Jupyter kernel installed in the Python environment (for code execution)
