# 11 — Project Structure

## High-level Overview

The project structure follows the **separation of concerns** principle across three layers: the domain (what the system does), the LangGraph layer (how it is orchestrated), and the infrastructure layer (how it is exposed and configured).

```
eda-agent/
│
├── README.md
├── pyproject.toml                  # Project dependencies and configuration
├── .env.example                    # Environment variable template
│
├── docs/                           # Documentation (this repository)
│
├── eda_agent/                      # Main Python package
│   │
│   ├── __init__.py
│   ├── config.py                   # EDAConfig (Pydantic BaseSettings)
│   ├── llm_factory.py              # get_llm() → BaseChatModel
│   ├── exceptions.py               # Domain-specific custom exceptions
│   │
│   ├── ingestion/                  # Data ingestion layer
│   │   ├── __init__.py
│   │   ├── loader.py               # load_file() → DataFrame
│   │   ├── profiler.py             # build_dataset_context() → DatasetContext
│   │   ├── semantic_classifier.py  # Column semantic classification
│   │   └── issue_detector.py       # Automatic issue detection
│   │
│   ├── models/                     # Pydantic domain models
│   │   ├── __init__.py
│   │   ├── dataset.py              # DatasetContext, ColumnInfo, BasicStats
│   │   ├── plan.py                 # EDAplan, EDAStep
│   │   ├── notebook.py             # NotebookCell, CellOutput
│   │   ├── execution.py            # ExecutionSummary, ExecutionResult
│   │   ├── critic.py               # CriticReview
│   │   └── session.py              # SessionMetadata, SessionStatus
│   │
│   ├── graph/                      # LangGraph layer
│   │   │
│   │   ├── parent/                 # Parent graph (Supervisor)
│   │   │   ├── __init__.py
│   │   │   ├── state.py            # EDAState TypedDict
│   │   │   ├── supervisor.py       # Parent graph definition
│   │   │   ├── router.py           # Conditional-edge routing logic
│   │   │   └── assembler.py        # Assembler node (builds the .ipynb)
│   │   │
│   │   └── subgraphs/
│   │       │
│   │       ├── planner/
│   │       │   ├── __init__.py
│   │       │   ├── state.py        # PlannerState
│   │       │   ├── graph.py        # Planner subgraph definition
│   │       │   ├── nodes.py        # analyze_dataset, generate_plan, validate_plan
│   │       │   └── prompts.py      # Planner system prompt and templates
│   │       │
│   │       ├── eda_loop/
│   │       │   ├── __init__.py
│   │       │   ├── state.py        # EDALoopState
│   │       │   ├── graph.py        # EDA Loop subgraph definition
│   │       │   ├── coder.py        # CODER node
│   │       │   ├── executor.py     # EXECUTOR node (no LLM)
│   │       │   ├── observer.py     # OBSERVER node
│   │       │   ├── findings_writer.py  # FINDINGS_WRITER node (no LLM)
│   │       │   └── prompts.py      # Coder and Observer prompts
│   │       │
│   │       ├── critic/
│   │       │   ├── __init__.py
│   │       │   ├── state.py        # CriticState
│   │       │   ├── graph.py        # Critic subgraph definition
│   │       │   ├── nodes.py        # read_section, evaluate_section
│   │       │   └── prompts.py      # Critic system prompt
│   │       │
│   │       └── conversation/
│   │           ├── __init__.py
│   │           ├── state.py        # ConversationState
│   │           ├── graph.py        # Conversation subgraph definition
│   │           ├── intent_classifier.py   # Intent classification
│   │           ├── handlers.py     # data_question, plan_modifier, general_responder
│   │           └── prompts.py      # Handler prompts
│   │
│   ├── tools/                      # Tools exposed to agents
│   │   ├── __init__.py
│   │   ├── kernel.py               # IPython kernel wrapper (sandbox)
│   │   ├── notebook_tools.py       # get_notebook_state, get_notebook_section
│   │   ├── data_tools.py           # get_column_stats, get_column_info, get_dtype_info
│   │   └── plan_tools.py           # add_step_to_plan, remove_step_from_plan
│   │
│   ├── memory/                     # Memory and context management
│   │   ├── __init__.py
│   │   ├── summarizer.py           # Message summarization logic
│   │   ├── history_filter.py       # Execution-history filtering for the Coder
│   │   └── context_builder.py      # Context construction for each agent
│   │
│   ├── checkpointing/              # Checkpointer setup
│   │   ├── __init__.py
│   │   └── setup.py                # Factory: MemorySaver (tests) / SqliteSaver (dev + prod)
│   │
│   ├── observability/              # Logging, tracing, metrics
│   │   ├── __init__.py
│   │   ├── callbacks.py            # EDAErrorCallbackHandler, EDAMetricsCallbackHandler
│   │   ├── logging_config.py       # Structured JSON logging configuration
│   │   ├── langsmith_config.py     # LangSmith setup
│   │   └── metrics.py              # Prometheus metrics definitions
│   │
│   └── api/                        # FastAPI layer
│       ├── __init__.py
│       ├── main.py                 # FastAPI app, middleware, lifespan
│       ├── dependencies.py         # FastAPI dependencies (checkpointer, kernel, etc.)
│       ├── schemas.py              # Pydantic request/response models
│       ├── event_translator.py     # LangGraph events → SSE events mapping
│       └── routes/
│           ├── __init__.py
│           ├── sessions.py         # POST/GET/DELETE /sessions
│           ├── stream.py           # GET /sessions/{id}/stream (SSE)
│           ├── hitl.py             # GET/POST /sessions/{id}/interrupt, /resume
│           ├── download.py         # GET /sessions/{id}/notebook
│           └── health.py           # GET /health, /health/kernel, /metrics
│
├── tests/                          # Test suite
│   ├── __init__.py
│   ├── conftest.py                 # Shared fixtures (mock LLM, test datasets)
│   ├── unit/
│   │   ├── test_ingestion.py
│   │   ├── test_profiler.py
│   │   ├── test_planner.py
│   │   ├── test_coder.py
│   │   ├── test_observer.py
│   │   ├── test_critic.py
│   │   └── test_assembler.py
│   ├── integration/
│   │   ├── test_eda_loop.py        # Tests the CODER→EXECUTOR→OBSERVER loop
│   │   ├── test_full_pipeline.py   # End-to-end test on a test dataset
│   │   └── test_resume.py          # Tests the resume mechanism
│   └── fixtures/
│       ├── sample_datasets/        # Small test datasets
│       └── expected_outputs/       # Expected outputs for integration tests
│
└── scripts/
    ├── run_dev.sh                  # Development startup (with hot reload)
    └── run_cli.py                  # CLI entry point
```

---

## Responsibilities of Key Modules

### `config.py`

Centralizes all system configuration. No values should be hardcoded elsewhere in the codebase. Every configurable parameter must have a sensible default and a clear description.

Separating base configuration (`EDAConfig`) from profiles (dev/prod) enables different behavior without duplicating code.

### `llm_factory.py`

The only place where LLM models are instantiated. No other module should directly import `ChatOpenAI`, `ChatAnthropic`, etc. This ensures switching providers requires changes in a single file.

The factory also builds the model with the correct retry configuration for API calls (exponential backoff, max retries, retriable errors).

### `graph/parent/supervisor.py`

The core orchestration entry point. It defines parent-graph nodes, conditional edges, and when each subgraph is invoked. It should be as readable as possible: readers should understand the full flow without opening other files.

Conditional edge functions (`router.py`) are extracted into a separate file to keep `supervisor.py` clean.

### `tools/kernel.py`

Wrapper around the IPython kernel. It exposes a simple interface (`execute(code) → ExecutionResult`) while hiding ZMQ communication complexity via `jupyter_client`. It includes timeout handling, kernel restart, and conversion of outputs into serializable formats.

### `api/event_translator.py`

Maps LangGraph event formats to SSE formats. This file encapsulates all transformation logic and must be well-tested: it is the backend/frontend contract.

### `memory/context_builder.py`

Builds agent-specific context before each LLM invocation. It takes global state and produces a dictionary of variables to inject into prompt templates. Centralizing this logic avoids duplication and guarantees consistency.

---

## Extensibility Principles

### Adding a new analysis type

1. Add the new type to `models/plan.py` (`analysis_type` enum)
2. Add inclusion rules to the Planner prompt in `subgraphs/planner/prompts.py`
3. Add evaluation criteria to the Critic prompt in `subgraphs/critic/prompts.py` (if needed)
4. Add any dedicated tools in `tools/`

### Adding a new LLM provider

1. Add a new case in `llm_factory.py`
2. Add the provider to the enum in `config.py`
3. Verify that the provider supports `with_structured_output` (most do)

### Adding a new input file type

1. Add a loader in `ingestion/loader.py`
2. Ensure the profiler (`ingestion/profiler.py`) works with the new type
3. Add the new type to validators in the `POST /sessions` endpoint
