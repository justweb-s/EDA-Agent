# 04 — Agents and Subgraphs

This document describes each system agent in detail: its responsibility, its internal graph, the available tools, the prompting strategy, and the input/output contracts with the parent.

---

## Agent 1: PLANNER

### Responsibility

The Planner has a single task: analyze the `DatasetContext` and produce a structured, adaptive, prioritized EDA plan. It does not execute code, does not perform analysis itself, and does not communicate with the user. It produces a structured artifact (a list of `EDAStep`) that drives the rest of the analysis.

### Internal subgraph structure

The Planner is a simple graph with three sequential nodes:

**`ANALYZE_DATASET`**: reads the `DatasetContext` and produces an initial analysis of dataset characteristics (variable types, obvious issues, special characteristics). This is not the plan yet—it is the "understanding" that informs the plan.

**`GENERATE_PLAN`**: based on the initial analysis, generates the full plan as a list of `EDAStep`. It uses `with_structured_output(EDAplan)` to guarantee structure.

**`VALIDATE_PLAN`**: a Python node (not an LLM node) that validates the plan: checks for circular dependencies, verifies referenced columns actually exist, and ensures there are no duplicate steps. If validation fails, the node either fixes the plan automatically or, for severe errors, returns to `GENERATE_PLAN` with specific guidance.

### Available tools

The Planner has no execution tools. It reasons exclusively over the pre-built `DatasetContext`.

### Prompting strategy

The Planner system prompt includes:

The `DatasetContext` serialized in a readable form (schema, types, statistics, samples, detected issues).

An explicit list of **standard EDA sections** to consider: data quality (nulls, duplicates, types), overview (shape, memory), univariate analysis (distributions, summary stats), bivariate analysis (correlations, scatter), multivariate analysis (heatmaps, PCA when applicable), outlier detection, feature-specific analysis.

**Explicit adaptivity rules**: if the dataset has only numeric columns, skip categorical analysis; if it has fewer than 5 numeric columns, skip PCA; if all columns have 0 nulls, skip the missing-values section; if a column name resembles "target" or "label", add feature–target relationship analysis.

A granularity constraint: each `EDAStep` must be small enough to require at most ~10–15 lines of code.

### Input from parent

- `dataset_context`: the complete `DatasetContext`
- `messages`: optional additional user instructions (e.g., "focus on time-series variables")

### Output to parent

- `eda_plan`: validated list of `EDAStep`

---

## Agent 2: EDA LOOP

### Responsibility

The EDA Loop is the heart of the system. For each `EDAStep` in the plan, it executes the full cycle: generate code → run it → observe output → decide whether the step is complete or needs a retry. Its outputs are notebook cells and the step's semantic findings.

### Internal subgraph structure

```
[CODER] ──▶ [EXECUTOR] ──▶ [OBSERVER]
   ▲                            │
   │     (retry with feedback)  │ verdict: retry
   └────────────────────────────┘
                                │ verdict: success
                                ▼
                           [FINDINGS_WRITER]
                                │
                                ▼
                           Output to parent
```

#### CODER node

The Coder is an LLM node that generates Python code for the current `EDAStep`.

**Inputs**: the current step (title, description, target columns, analysis type), the `DatasetContext`, the `execution_history` of previous steps (semantic findings only, not raw code), and—in case of retry—the previous code plus the error message.

**Output**: a `CodeCell` object with `code: str` (Python code) and `expected_output_description: str` (what the Observer should expect to see).

**Explicit prompt constraints**:

- The DataFrame is always named `df` and is already loaded (available in the kernel from the start)
- Do not use `plt.show()` — save plots via `plt.savefig(buf, format='png')` and `buf = io.BytesIO()`
- Do not load files — `df` is already available
- Do not install libraries — only pandas, numpy, matplotlib, seaborn, scipy, sklearn
- Write the minimum code necessary for this specific step
- End the code with an explicit output statement (print, display, return value)

**Retry behavior**: when the Coder receives an error, the prompt includes the full error, the previous code, and explicit instructions to avoid repeating the same mistake.

#### EXECUTOR node

The Executor is not an LLM node: it is a Python node that sends code to the IPython kernel and captures output.

**Kernel communication**: uses `jupyter_client` to send messages to the kernel via ZMQ. The kernel preserves Python state across executions—variables defined in one cell are available in subsequent ones.

**Captured outputs**: stdout, stderr, display_data (images, HTML), execute_result (last expression value). Images are captured as PNG bytes and converted to base64 for serialization in state.

**Timeouts**: each execution has a configurable timeout (default: 60 seconds). A timeout is treated as an error and passed to the Observer with a specific message.

**Isolation**: the kernel is a separate process. A `sys.exit()` or a crash in generated code does not crash the main process. If the kernel crashes (rare), it is automatically restarted and the DataFrame `df` is reloaded.

#### OBSERVER node

The Observer is an LLM node that analyzes execution output and decides the next action.

**Inputs**: executed code, full output (stdout, stderr, images), and the Coder's `expected_output_description`.

**Output**: an `ObserverVerdict` object with:

- `verdict`: `"success"`, `"retry"`, or `"fatal_error"`
- `findings_description`: natural-language description of what was found (used for execution history)
- `key_statistics`: dictionary of key numeric stats extracted from the output
- `error_analysis`: error analysis (only for `"retry"`)
- `retry_hint`: specific hint for the Coder on how to fix the code (only for `"retry"`)

**Evaluation criteria**:

- `"success"`: code produced sensible output consistent with the expected output
- `"retry"`: code produced an error, or produced empty/non-informative output
- `"fatal_error"`: the error cannot be fixed by changing code (e.g., a non-existent column—should be prevented by `DatasetContext`)

#### FINDINGS_WRITER node

A Python node (not an LLM node) that produces the `ExecutionSummary` to append to the parent's `execution_history`. It aggregates Observer information into a standardized format.

### Tools available to the EDA Loop

EDA Loop tools are available to the CODER to query specific information before writing code:

**`get_column_stats(column_name: str)`**: returns detailed statistics for a specific column. Useful when the Coder needs more granular information than what is available in `DatasetContext`.

**`get_dtype_info(column_name: str)`**: returns the Pandas dtype and a value sample. Useful to decide which analysis type is appropriate.

Note: these tools read directly from the DataFrame already loaded in the kernel, so they are always consistent with the current data state.

### Input from parent

- `current_step`: the current `EDAStep` to execute
- `dataset_context`: the `DatasetContext`
- `execution_history`: findings history from previous steps

### Output to parent

- `notebook_cells`: list of new cells to add (typically: 1 intro markdown cell + 1 code cell with output)
- `execution_summary`: the `ExecutionSummary` with step findings

---

## Agent 3: CRITIC

### Responsibility

The Critic evaluates the quality of each completed EDA section and decides whether deeper work is required. It acts as an "external reviewer" that reads the produced notebook and checks that the analysis is complete, accurate, and not superficial.

### Internal subgraph structure

The Critic is a simple graph with two nodes:

**`READ_SECTION`**: collects the section cells to analyze and organizes them into an LLM-friendly format.

**`EVALUATE_SECTION`**: produces `CriticReview` using `with_structured_output`.

### Prompting strategy

The Critic system prompt is deliberately different from other agents: it is strict and quality-focused. It includes:

A description of what constitutes **good EDA** for each section type (distributions, correlations, outliers, etc.). For example, for distributions: "a histogram is not enough—include summary statistics, normality checks, multimodality detection, and outlier analysis".

A list of **red flags** to look for: plots without titles/axis labels, statistics with no interpretation, analyses that ignore nulls, correlations reported without significance considerations.

**Proportionality instructions**: the Critic must not request excessive or irrelevant analysis. Its quality bar should match an experienced but pragmatic data scientist.

### Input from parent

- `notebook_cells`: cells for the current section
- `dataset_context`: to verify that all relevant columns were analyzed
- `section_name`: section name (to apply the correct evaluation criteria)

### Output to parent

- `critic_feedback`: the full `CriticReview`

---

## Agent 4: CONVERSATION

### Responsibility

The Conversation agent handles user interaction during and after analysis. It classifies user intent, selects the appropriate action, and executes ad-hoc code when required to answer dataset-specific questions.

### Internal subgraph structure

```
[INTENT_CLASSIFIER] ──▶ routing ──▶ [DATA_QUESTION_HANDLER]
                                 ──▶ [PLAN_MODIFIER]
                                 ──▶ [GENERAL_RESPONDER]
```

#### INTENT_CLASSIFIER node

Classifies the user message into one of the following categories:

- `data_question`: a question that requires analyzing the data (e.g., "how many customers are older than 50?")
- `analysis_request`: a request for an analysis not in the plan (e.g., "can you do segmentation?")
- `plan_modification`: the user wants to modify the plan (e.g., "skip correlation analysis")
- `step_rerun`: the user wants to rerun a previous step (e.g., "redo the distribution plot with more bins")
- `general_question`: a general question that does not require code (e.g., "what does kurtosis mean?")

#### DATA_QUESTION_HANDLER node

For questions that require code execution, this node generates and executes ad-hoc code (without adding it to the notebook) and then produces a textual answer based on the output.

Ad-hoc code runs on the same IPython kernel as the main analysis, so it has access to `df` and all variables defined during analysis.

#### PLAN_MODIFIER node

For plan modification requests, this node updates the `eda_plan` list in state (add steps, remove steps, modify existing steps).

#### GENERAL_RESPONDER node

For general questions, it answers directly without running code.

### Available tools

**`get_notebook_state()`**: returns a summary of the current notebook (completed sections, key findings, produced charts).

**`execute_adhoc_code(code: str)`**: executes code in the kernel and returns output. Used by DATA_QUESTION_HANDLER.

**`add_step_to_plan(step: EDAStep)`**: adds a step to the current plan.

**`remove_step_from_plan(step_id: str)`**: removes a step from the plan.

**`get_column_info(column_name: str)`**: detailed information about a column.

### Input from parent

- `messages`: full message history (including the new user message)
- `dataset_context`: to answer data questions
- `eda_plan`: current plan (for modifications)
- `notebook_cells`: current notebook state

### Output to parent

- `messages`: agent response as an `AIMessage`
- `eda_plan` (optional): modified plan, if requested

---

## Assembler (Python node, not an agent)

The Assembler is not an LLM agent: it is a Python node that collects all accumulated `notebook_cells` and assembles a valid `.ipynb` file using `nbformat`.

### Assembly process

1. Create a new notebook via `nbformat.v4.new_notebook()` with appropriate metadata (Python 3 kernel, analyzed dataset metadata, timestamp)
2. Add a header Markdown cell with analysis title, file name, date, and the LLM model used
3. Add a setup code cell with standard imports
4. Iterate over `notebook_cells` in the order they were added, converting each `NotebookCell` into nbformat
5. For code cells, include captured outputs in nbformat format (`text/plain`, `image/png`, etc.)
6. Write the notebook to disk and save the path in `final_notebook_path`

### Why the produced notebook is already executed

An important characteristic: the produced notebook is not just a notebook with code—it is a notebook with code **and outputs**. This is because every cell is actually executed during generation and outputs are captured. Users can open the notebook and immediately see results without re-running it.

The notebook can still be re-run by the user (cell by cell or all at once) to verify reproducibility.
