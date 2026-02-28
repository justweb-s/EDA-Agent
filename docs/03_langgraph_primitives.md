# 03 — LangGraph Primitives

This document describes, in detail, the LangGraph primitives used by the system, the rationale behind each choice, and the design implications.

---

## `StateGraph`

`StateGraph` is LangGraph's fundamental primitive for building stateful graphs. Unlike a simple DAG, a `StateGraph` maintains a shared state that is updated at each node, persisted by the checkpointer, and accessible to all graph nodes.

### How it is used

Both the parent graph (Supervisor) and each subgraph are instances of `StateGraph`. The key difference is that the parent graph is compiled with a checkpointer (for persistence), while subgraphs are compiled without one (their state is transient and exists only during subgraph execution).

### Compilation

The graph is compiled once at application startup via `.compile(checkpointer=checkpointer)`. The compiled graph is immutable and thread-safe: it can be invoked concurrently with different `thread_id` values. In the single-user scenario this is not critical, but it is best practice to avoid recompiling the graph on every request.

### Conditional edges

The Supervisor uses `add_conditional_edges()` to implement dynamic routing. A conditional edge is a Python function that receives the current state and returns the name of the next node.

The main branching points in the Supervisor are:

**After the Planner**: if `hitl_enabled=True`, route to `HITL_PLAN_APPROVAL` which calls `interrupt()`; otherwise route directly to `EDA_STEP_ROUTER`.

**After each completed EDA step**: `EDA_STEP_ROUTER` reads `current_step_index` and `eda_plan` and decides whether another step needs to run (route to `EDA_LOOP`), whether the section is completed (route to `CRITIC`), or whether all steps are finished (route to `ASSEMBLER`).

**After the Critic**: if `verdict == "ok"`, proceed to the next group of steps; if `verdict == "needs_improvement"`, append corrective steps to the plan and return to `EDA_STEP_ROUTER`; if the maximum number of Critic iterations has been reached, proceed anyway to the Assembler.

**After the Assembler**: if `mode == "chat"`, route to `WAIT_FOR_USER` (interrupt); if `mode == "auto"`, route to `END`.

---

## Subgraphs

A subgraph is a compiled `StateGraph` mounted as a single node in the parent graph. From the parent's perspective, it is an opaque node: it receives input, does work, and returns output.

### Input/output contract

Communication between parent and subgraph happens through an explicit contract:

- The parent passes a subset of its state as input to the subgraph
- The subgraph returns updates to the parent state as output
- The mapping between the two states' fields is declared when mounting the subgraph into the parent

This design ensures subgraphs are **reusable and testable in isolation**: they can be invoked with a test input without creating the full parent graph.

### Private subgraph state

Subgraphs may have additional state fields that do not exist in the parent state. These fields are used for internal coordination (e.g., `retry_messages` in the EDA Loop, `draft_plan` in the Planner) and are never exposed to the parent.

### Nesting and error propagation

Unhandled errors inside a subgraph propagate to the parent as normal exceptions. The parent is responsible for handling them either via conditional-edge logic or via node-level error handling.

---

## Checkpointing

Checkpointing is the mechanism that makes the system **durable**: after each super-step (each completed node), LangGraph serializes the full state and saves it to the configured persistence backend.

### Available backends

**`MemorySaver`**: stores state in Python memory. Used in tests and for throwaway sessions. State is lost on process restart.

**`SqliteSaver` / `AsyncSqliteSaver`**: stores state in a local SQLite database. This is the chosen backend for both development and production: no external dependencies, reliable, and perfectly adequate for a single-user system. It is installed via the separate package `langgraph-checkpoint-sqlite`.

### `thread_id`

`thread_id` is the cursor that identifies a specific session in the checkpointer. It is a UUID generated when a session starts and passed to each graph invocation via config:

```
config = {"configurable": {"thread_id": session_id}}
```

Reusing the same `thread_id` means resuming the session from the saved state. Using a new `thread_id` means starting a fresh session.

### Crash resume

If the system crashes during step 7 of 12, the checkpointer has saved state after step 6. To resume, invoke the graph again with the same `thread_id`: LangGraph loads the saved state and restarts from step 7 without re-running prior steps.

In the UI, this becomes a list of sessions with their status (`in_progress`, `completed`, `failed`) and a "Resume" button for each non-completed session.

### Checkpoint granularity

LangGraph saves state after each **super-step**, corresponding to the full execution of a node (or a group of parallel nodes). In this system, every Supervisor node and every subgraph node is a super-step.

Granularity is therefore fine: if the system crashes mid EDA Loop subgraph invocation (e.g., while executing code), resume will restart from the beginning of that subgraph invocation, not from the beginning of the entire analysis.

---

## Interrupts

`interrupt()` is LangGraph's native mechanism for **Human-in-the-Loop**. When called inside a node, the graph immediately freezes, serializes state to the checkpointer, and returns to the caller an object containing the interrupt payload.

Execution remains suspended—potentially for hours or days—until the caller invokes the graph again with a `Command(resume=value)`.

### Dynamic interrupts vs static breakpoints

LangGraph offers two mechanisms: static breakpoints (defined at compile-time via `interrupt_before` or `interrupt_after` on specific nodes) and dynamic interrupts (calling `interrupt()` in node code, conditionally at runtime).

This system uses **dynamic interrupts** because the decision to interrupt depends on the runtime `hitl_enabled` flag, not compile-time configuration.

### Interrupt payload

`interrupt()` accepts any JSON-serializable value as payload. In this system, the payload always includes:

- `interrupt_type`: type of interrupt (`"plan_approval"`, `"critic_review"`, `"section_completed"`)
- `data`: the data to show the user (EDA plan, Critic review, current notebook state)
- `available_actions`: list of actions the user can take

This payload is exposed by the SSE API as an event of type `"hitl_interrupt"` to the frontend.

### `Command` resume

Resuming from an interrupt is done via `Command(resume=value)`, passed instead of the normal input when invoking the graph again. The value becomes the return value of the suspended `interrupt()` call in the node.

The resume value includes:

- For `plan_approval`: the approved plan (possibly modified by the user)
- For `critic_review`: `"accept"` or `"request_improvement"` plus optional notes
- For `section_completed`: the user's message

---

## `MessagesAnnotation`

`MessagesAnnotation` is a built-in LangGraph annotation that manages the message list intelligently. It behaves like `add_messages` but adds:

- Automatic append of new messages
- Updating existing messages (by ID) instead of duplicating
- Correct handling of different message types (`HumanMessage`, `AIMessage`, `SystemMessage`, `ToolMessage`)

In the parent state, the `messages` field uses `MessagesAnnotation` via inheritance from `MessagesState` (a LangGraph shortcut that already includes the correctly annotated `messages` field).

---

## Streaming

LangGraph provides a multi-layer streaming system, used extensively in this project. Full details are in [Streaming](./06_streaming.md); this section focuses on the primitives.

**`.astream(input, config, stream_mode)`**: async method that turns a graph invocation into an async iterator of events. `stream_mode` accepts a list of modes that can be enabled simultaneously.

**`stream_mode="updates"`**: emits an event whenever a node updates state. The event includes the node name and the updated values. This is the most useful mode for the UI: each new notebook cell produces an update.

**`stream_mode="messages"`**: emits LLM tokens as they are generated. This enables showing the Coder code or Critic commentary token-by-token in the UI.

**`stream_mode="custom"`**: allows nodes to emit arbitrary events via `StreamWriter`. Used for progress signals such as "executing code", "retry #2", "step skipped due to error".

**`subgraphs=True`**: additional `.astream()` parameter that includes events emitted inside subgraphs, not only in the parent. This is essential because most work happens within subgraphs.

---

## `with_structured_output`

`with_structured_output(PydanticModel)` is a `BaseChatModel` method that instructs the model to always return structured output conforming to a Pydantic schema, using the underlying provider's function-calling/structured-output mechanism.

This system uses it systematically for all nodes that produce outputs consumed programmatically by other nodes:

- The Planner uses `with_structured_output(EDAplan)` to guarantee the plan is a validated list of `EDAStep`
- The Coder uses `with_structured_output(CodeCell)` to separate code from its expected-output description
- The Critic uses `with_structured_output(CriticReview)` to guarantee the verdict is one of the expected values
- The Conversation agent uses `with_structured_output(ConversationResponse)` to classify intent and structure responses

The alternative—parsing free-form LLM text—is fragile, hard to test, and produces errors that are difficult to debug. `with_structured_output` shifts validation responsibility to the provider layer (via function calling) and to Pydantic, drastically reducing parsing errors.

---

## Parallel node execution

LangGraph supports parallel node execution via **fan-out / fan-in**: if a node has edges to two nodes with no mutual dependency, LangGraph can run them in parallel within the same super-step.

This could be used in the future to execute independent analyses in parallel (e.g., distribution analysis for column A and column B at the same time). The current design intentionally avoids parallelization for simplicity and to prevent race conditions around the IPython kernel state (which is single-threaded by nature).

---

## Send API (dynamic fan-out)

LangGraph's `Send` API allows dynamically creating parallel instances of a subgraph with different inputs. Instead of iterating sequentially over a list of steps, one could use `Send` to launch all steps in a section in parallel.

As with parallel execution above, this feature is not used in the current design in order to keep the IPython kernel execution serial. It is documented for completeness and future evolution.
