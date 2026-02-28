# 07 — Human-in-the-Loop

## HITL Philosophy in the System

Human-in-the-Loop is not designed as a safety mechanism (the system is not meant to execute dangerous code), but as a tool for **quality and direction**. The goal is to let the user influence the analysis at key moments without having to monitor every single step.

HITL is **configurable by design**: in fully automatic single-shot mode it is completely disabled; in chat mode it is always active; in explicit HITL mode it is enabled only at predefined touchpoints.

---

## Core Mechanism: `interrupt()` and `Command`

LangGraph implements HITL via the `interrupt()` function, which can be called inside any node conditionally at runtime.

When `interrupt()` is called:

1. Node execution stops immediately
2. LangGraph saves the full state to the checkpointer (including the exact interruption point)
3. The graph returns an object with interrupt details in the `__interrupt__` field
4. The graph remains in a "suspended" state and can wait indefinitely

To resume, the caller invokes the graph again with the same `thread_id`, passing `Command(resume=value)` instead of the normal input. The `value` becomes the return value of `interrupt()`, and execution continues from that point.

### Dynamic interrupts vs static breakpoints

The system uses **dynamic interrupts** (calling `interrupt()` in node code) instead of static breakpoints (`interrupt_before`/`interrupt_after` configured at compile time). The reason is that the interruption decision depends on `hitl_enabled`, which is a runtime user choice.

---

## The Four HITL Touchpoints

### Touchpoint 1: Plan Approval

**When**: after the Planner generates the EDA plan, before any code execution begins.

**What the user sees**: the complete plan as a list of sections and steps, with a human-readable description of each step, involved columns, and an estimate of the number of operations.

**Available user actions**:

- `approve`: the plan is OK, proceed
- `modify_plan`: the user can remove steps, add steps, change ordering, or add step-specific instructions
- `regenerate`: the plan is not OK; generate a new plan (optionally with guidance on what to change)

**Resume value**: the plan object (possibly modified) or a regenerate instruction.

**Rationale**: the plan is the analysis "contract". Having the user approve it before starting prevents realizing after 10 minutes that the analysis went in the wrong direction.

### Touchpoint 2: Critic Review

**When**: every time the Critic emits `verdict: needs_improvement`.

**What the user sees**: the Critic review with quality score, missing analyses, shallow interpretations, and the list of proposed corrective steps.

**Available user actions**:

- `accept_critic`: the Critic is right; run the proposed corrective steps
- `reject_critic`: current analysis is acceptable; proceed without corrections
- `partial_accept`: accept only some corrective steps (the user specifies which)
- `custom_correction`: specify corrections different from the Critic's proposal

**Resume value**: the user's decision and any custom steps.

**Rationale**: the Critic may be too strict or propose irrelevant analysis for the specific use case. The user is the best judge of the required quality level.

### Touchpoint 3: End of Section (chat mode)

**When**: in chat mode, after each EDA section is completed and approved by the Critic.

**What the user sees**: a section summary with key findings, produced charts, and counts of completed/failed steps.

**Available user actions**:

- `continue_analysis`: continue with the next section in the plan
- `ask_question`: ask a question about the data or intermediate results
- `add_analysis`: add a specific analysis to the next section
- `skip_to_end`: skip remaining sections and produce a notebook with what has been generated so far
- `end_session`: analysis is complete; produce the final notebook

**Rationale**: in chat mode, the user may want specific analyses or questions on intermediate results. Pausing after each section (not after each single step) is a good compromise between autonomy and control.

### Touchpoint 4: Critical Step (optional, configurable)

**When**: optionally, before executing steps marked as `is_critical=True` in the plan. The Planner can set this flag for high-impact steps (e.g., outlier removal on small datasets, irreversible transformations).

**Note**: in the current system the Planner does not modify the original DataFrame—it only produces analysis. This touchpoint is therefore not very relevant right now, but it is documented for completeness and for potential future evolutions that may include preprocessing suggestions.

---

## Handling Interrupts in the UI

### Receiving the interrupt

When the backend emits an SSE event of type `hitl_interrupt`, the UI:

1. Freezes the in-progress analysis view (no new elements are appended)
2. Shows a modal or side panel with interrupt information
3. Renders available actions as buttons
4. Waits for user input before proceeding

### Sending the resume

When the user selects an action (with optional modifications), the UI sends a request to `POST /sessions/{session_id}/resume` with payload:

```json
{
  "interrupt_id": "uuid-of-the-interrupt",
  "action": "approve",
  "data": { ... optional modifications ... }
}
```

The backend builds the appropriate `Command(resume=value)` and re-invokes the graph.

### Interrupt timeout

An interrupt can wait indefinitely because LangGraph persists state. However, in production it is sensible to configure a timeout (e.g., 24 hours) after which the session is marked as `abandoned` and resources are released.

The timeout does not cause data loss—the checkpoint is still available—but the session is no longer shown as "active" in the UI.

---

## HITL Configuration

HITL activation is controlled by three separate mechanisms:

**`hitl_enabled: bool`** in session state: controls whether interrupts are actually executed at runtime. When `False`, nodes containing `interrupt()` calls skip them and proceed automatically.

**`hitl_mode`** in session configuration: `"none"` (no interrupts), `"plan_only"` (plan approval only), `"full"` (all touchpoints), `"custom"` (explicit list of touchpoints to enable).

**`EDAConfig.hitl_touchpoints: list[str]`**: list of enabled touchpoints when `hitl_mode == "custom"`. Possible values: `"plan_approval"`, `"critic_review"`, `"section_completed"`, `"critical_step"`.

---

## HITL and Operating Modes

| Mode | hitl_enabled | Active touchpoints |
|------|--------------|--------------------|
| automatic single-shot | False | None |
| single-shot with HITL | True | plan_approval, critic_review |
| interactive chat | True | section_completed (always), plan_approval and critic_review (optional) |
