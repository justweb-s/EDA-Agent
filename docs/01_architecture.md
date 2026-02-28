# 01 — Overall Architecture

## Design Philosophy

The system architecture is built around three principles that guide every technical decision.

The first is **separation between planning and execution**. A system that generates and immediately executes large blocks of code makes two mistakes: it produces longer, harder-to-debug code, and it cannot adapt to intermediate results. In this system, each EDA step is a full cycle of plan → generate → execute → observe. The next step is planned only after seeing the output of the current step.

The second is **agent specialization**. A single agent that does everything (planning, writing code, quality assessment, conversation) has to keep too much context and tends to perform worse across the board. Four specialized agents, each with an optimized prompt and a constrained toolset, produce significantly better results.

The third is **flow durability**. A real EDA on a real dataset can take dozens of steps and several minutes. The system must be resilient to crashes, timeouts, and user interruptions without losing completed work. This is provided by LangGraph's native checkpointing.

---

## Architectural Pattern: Supervisor + Specialized Subgraphs

The chosen pattern is a variant of the **Supervisor Pattern** described by LangGraph for multi-agent systems.

The **Supervisor** is a parent graph that owns the global session state and decides, via conditional edges, which subgraph to invoke and when. It does not perform advanced reasoning: its job is orchestration, not analysis.

The **Subgraphs** are autonomous LangGraph graphs, compiled separately and mounted as nodes inside the parent. Each receives a slice of the global state as input, operates on its private state, and returns updates back to the parent state as output. The parent does not know the internal details of each subgraph; it only knows the input/output contract.

This structure brings three practical benefits:

1. Each subgraph can be developed, tested, and debugged in isolation.
2. Errors inside a subgraph do not automatically cascade into the whole system: the parent can handle them with its own logic.
3. The system is easily extensible: adding a specialized agent means adding a new subgraph and connecting it to the Supervisor.

---

## Parent Graph Diagram

```
                              ┌─────────────────────────────────────────────────┐
                              │              PARENT GRAPH (Supervisor)           │
                              │                                                   │
  Input ──▶ [DATA INGESTION] ─▶ [PLANNER] ──▶ [EDA LOOP] ──▶ [CRITIC] ──▶ ...  │
              (pure Python)      subgraph      subgraph       subgraph            │
                              │      │             │               │               │
                              │      │    ◀────────┘  (feedback)  │               │
                              │      │                             │               │
                              │      ▼                             ▼               │
                              │  [HITL interrupt]           [updated plan]         │
                              │  (if enabled)                    │                 │
                              │                                   ▼                 │
                              │                            [EDA LOOP] (retry)      │
                              │                                    │               │
                              │                    all steps completed              │
                              │                                    │               │
                              │                                    ▼               │
                              │                            [ASSEMBLER] ──▶ Output  │
                              │                                    │               │
                              │                    (chat mode)     │               │
                              │                                    ▼               │
                              │                         [CONVERSATION] ◀── User   │
                              │                                    │               │
                              │                      ◀────────────┘ (modify)      │
                              └─────────────────────────────────────────────────┘
```

---

## Detailed Execution Flow

### Phase 1: Data Ingestion (pre-agent)

Before any LLM is invoked, the system loads the file and builds the `DatasetContext`. This phase is pure Python code, with no LLM calls. Its output—schema, types, statistics, sample—is serialized and injected into every agent's system prompt as the foundational context.

Building the `DatasetContext` before starting the graph is deliberate: it prevents the LLM from "discovering" the dataset token-by-token during analysis, and ensures every agent starts from a complete and coherent view of the data.

### Phase 2: Planning

The Planner subgraph receives the `DatasetContext` and produces a structured EDA plan as a list of `EDAStep`. The plan includes metadata for each step (analysis type, dependencies on previous steps, mandatory flags) that the Supervisor uses to manage execution order and skip decisions.

The plan is adaptive by design: the Planner has explicit instructions to avoid analyses that do not make sense for the specific dataset (e.g., numeric correlations on a dataset with only categorical variables).

If HITL is enabled, the Supervisor calls `interrupt()` after planning to present the plan to the user and wait for approval or modifications.

### Phase 3: EDA Loop

For each `EDAStep` in the plan, the Supervisor invokes the EDA Loop subgraph. The loop is the core of the system and implements the cycle CODER → EXECUTOR → OBSERVER.

The CODER generates code for the current step. The EXECUTOR runs it in the IPython kernel and captures the output. The OBSERVER analyzes the output, extracts semantic findings, and decides whether the step is complete or requires a retry.

The execution history seen by the CODER contains **semantic findings** from previous steps (not raw code), so the newly generated code can incorporate what has been discovered so far. This is the key mechanism behind adaptivity.

### Phase 4: Critic Review

When an EDA section (a logical group of steps, e.g. "distribution analysis") is completed, the Supervisor invokes the Critic subgraph. The Critic reads the cells produced for that section and outputs a structured evaluation: analysis quality, missing aspects, and shallow interpretations.

If the Critic emits `verdict: needs_improvement`, the Supervisor appends corrective steps to the plan and re-runs the loop. This cycle can repeat up to a configurable maximum number of Critic iterations to avoid infinite loops.

### Phase 5: Assembler

When all plan steps have been completed and accepted by the Critic, the Assembler node collects all accumulated cells from the state and assembles a valid `.ipynb` notebook using `nbformat`. The notebook includes both code cells (with captured execution outputs) and Markdown cells with interpretive comments.

### Phase 6: Conversation (optional)

In chat mode, after each completed section (or after the final Assembler), the Supervisor enters a waiting state via `interrupt()`, and control passes to the Conversation subgraph when user input arrives. The Conversation agent can answer questions, modify the plan, re-run steps, or execute ad-hoc analyses.

---

## Responsibilities of Each Component

| Component | Type | Primary responsibility | Uses an LLM? |
|-----------|------|------------------------|--------------|
| Data Ingestion | Python module | Load file, build `DatasetContext` | No |
| Supervisor | Parent graph | Orchestration, routing, HITL | No (conditional edges only) |
| Planner | Subgraph | Generate an adaptive plan | Yes |
| EDA Loop | Subgraph | Code → execute → observe loop | Yes (Coder + Observer) |
| Critic | Subgraph | Review quality of EDA sections | Yes |
| Conversation | Subgraph | Handle user input, ad-hoc analysis | Yes |
| Assembler | Python node | Build the final `.ipynb` notebook | No |
| IPython Kernel | Service | Execute Python code with persistent state | No |

---

## Key Design Decisions and Rationale

### Why LangGraph instead of a simple Python loop?

A standard Python loop does not provide: persistent state across steps, checkpointing for resume, native interrupt handling for HITL, streaming of state updates, or declarative conditional edges. LangGraph solves these problems natively and is production-tested.

### Why is the Critic a separate agent?

An external reviewer reading an already-written notebook has different biases compared to the agent that produced it. The Critic has a system prompt focused exclusively on analysis quality and completeness, not on code generation. This separation yields more critical and accurate reviews.

### Why does the execution history contain semantic findings instead of code?

Passing raw previous code to the CODER would quickly fill the context window with low-density information. Findings (e.g., "column X has 12% nulls, a right-skewed distribution, and 5 outliers beyond 3σ") are much more useful and compact than the code that produced them.

### Why an IPython kernel instead of direct `exec()`?

A separate IPython kernel provides: persistent Python state across cells (variables defined in step 2 are available in step 5, exactly like in a real notebook), isolation from the main process, timeout handling, and rich outputs (images, dataframes). Direct `exec()` in the main process provides none of these guarantees.

### Why SSE instead of WebSockets for streaming?

The communication pattern is one-way: the server sends updates to the browser, and the browser does not need continuous streaming uploads. SSE is simpler to manage, more reliable on disconnects (native auto-reconnect), and integrates naturally with FastAPI and LangGraph's `astream()`.
