# 12 — Configuration

## Configuration Philosophy

All configurable parameters are centralized in `EDAConfig`, a Pydantic `BaseSettings` model that reads values from environment variables (or a `.env` file). No values are hardcoded in application code.

Each parameter has:

- A **sensible default** that works out of the box in development
- A **description** that explains its purpose and the impact of changing it
- A **Pydantic validation** that prevents invalid values

---

## Environment Variables

### LLM provider

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LLM_PROVIDER` | str | `openai` | Default LLM provider (`openai`, `anthropic`, `ollama`, `groq`) |
| `LLM_MODEL` | str | `gpt-4o` | Default model for the selected provider |
| `OPENAI_API_KEY` | str | — | OpenAI API key (required if provider=openai) |
| `ANTHROPIC_API_KEY` | str | — | Anthropic API key (required if provider=anthropic) |
| `GROQ_API_KEY` | str | — | Groq API key (required if provider=groq) |
| `OLLAMA_BASE_URL` | str | `http://localhost:11434` | Base URL for a local Ollama server |
| `LLM_TEMPERATURE` | float | `0.1` | Temperature for all models (low for more deterministic output) |
| `LLM_MAX_RETRIES` | int | `3` | Max retries for LLM API errors |

### Database and persistence

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `ENVIRONMENT` | str | `development` | Current environment (`development`, `production`) |
| `CHECKPOINT_BACKEND` | str | `sqlite` | Checkpointer backend (`memory`, `sqlite`) |
| `SQLITE_PATH` | str | `./checkpoints.db` | Path to the SQLite file used in both development and production |
| `CHECKPOINT_TTL_HOURS` | int | `72` | Hours after which a checkpoint is automatically deleted (manual cleanup via script) |

### Code execution

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `KERNEL_EXECUTION_TIMEOUT` | int | `60` | Timeout in seconds for each code execution |
| `KERNEL_HEAVY_TIMEOUT` | int | `300` | Timeout for computationally heavy steps |
| `KERNEL_RESTART_ON_CRASH` | bool | `true` | Automatically restart the kernel on crash |
| `KERNEL_MAX_OUTPUT_SIZE_MB` | float | `5.0` | Max size of captured outputs per cell |

### Analysis parameters

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `MAX_STEP_RETRIES` | int | `3` | Max retries for failed EDA steps |
| `MAX_CRITIC_ITERATIONS` | int | `2` | Max Critic iterations per section before proceeding anyway |
| `MANDATORY_SECTIONS` | list[str] | `["data_quality", "overview"]` | Sections always included in the plan |
| `NULL_WARNING_THRESHOLD` | float | `0.05` | Null percentage threshold for warnings in `DatasetContext` |
| `NULL_ERROR_THRESHOLD` | float | `0.30` | Null percentage threshold for errors in `DatasetContext` |
| `OUTLIER_SIGMA_THRESHOLD` | float | `3.0` | Number of σ used to classify outliers |
| `HIGH_CARDINALITY_THRESHOLD` | int | `20` | Unique-value threshold used to classify "high cardinality" columns |

### Memory management

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `CONTEXT_WINDOW_WARNING_PCT` | float | `0.80` | Context window percentage beyond which summarization is triggered |
| `SUMMARIZATION_KEEP_LAST_N` | int | `10` | Number of most recent messages kept verbatim after summarization |
| `EXECUTION_HISTORY_MAX_SECTION_DETAIL` | int | `5` | Max recent steps of the current section passed in detail to the Coder |

### Human-in-the-Loop

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `HITL_DEFAULT_MODE` | str | `none` | Default HITL mode (`none`, `plan_only`, `full`, `custom`) |
| `HITL_TIMEOUT_HOURS` | int | `24` | Hours after which a HITL-suspended session is marked as `abandoned` |
| `HITL_TOUCHPOINTS` | list[str] | `["plan_approval", "critic_review"]` | Enabled touchpoints when `hitl_mode=custom` |

### Observability

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `LANGCHAIN_TRACING_V2` | bool | `false` | Enables LangSmith tracing |
| `LANGCHAIN_API_KEY` | str | — | LangSmith API key (required if tracing enabled) |
| `LANGCHAIN_PROJECT` | str | `eda-agent` | LangSmith project name |
| `LOG_LEVEL` | str | `INFO` | Log level (`DEBUG`, `INFO`, `WARNING`, `ERROR`) |
| `LOG_FORMAT` | str | `json` | Log format (`json`, `text`) |
| `ALERT_CHANNEL` | str | `log` | Alert channel (`log`, `email`, `slack`, `webhook`) |
| `ALERT_SLACK_WEBHOOK_URL` | str | — | Slack webhook URL (required if alert_channel=slack) |
| `ALERT_EMAIL_RECIPIENTS` | list[str] | `[]` | Email recipients for alerts |

### API

| Variable | Type | Default | Description |
|----------|------|---------|-------------|
| `API_HOST` | str | `0.0.0.0` | FastAPI server host |
| `API_PORT` | int | `8000` | FastAPI server port |
| `API_KEY` | str | — | API key for authentication (required in production) |
| `UPLOAD_DIR` | str | `./uploads` | Directory for uploaded files |
| `OUTPUT_DIR` | str | `./outputs` | Directory for produced notebooks |
| `MAX_UPLOAD_SIZE_MB` | float | `100.0` | Max upload size |
| `SSE_BUFFER_SIZE` | int | `100` | Max number of buffered SSE events per session |

---

## Configuration Profiles

### Development profile

```env
ENVIRONMENT=development
CHECKPOINT_BACKEND=sqlite
SQLITE_PATH=./checkpoints_dev.db
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o-mini          # Cheaper model for development
LLM_TEMPERATURE=0.1
LOG_LEVEL=DEBUG
LOG_FORMAT=text
LANGCHAIN_TRACING_V2=true       # Useful in development for debugging
HITL_DEFAULT_MODE=full          # Exercise all touchpoints
MAX_STEP_RETRIES=2              # Fail fast in development
KERNEL_EXECUTION_TIMEOUT=30     # Short timeout for fast tests
```

### Production profile

```env
ENVIRONMENT=production
CHECKPOINT_BACKEND=sqlite
SQLITE_PATH=/var/data/eda_agent/checkpoints.db   # Path on a persistent volume
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
LLM_TEMPERATURE=0.1
LOG_LEVEL=INFO
LOG_FORMAT=json
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=eda-agent-prod
ALERT_CHANNEL=slack
ALERT_SLACK_WEBHOOK_URL=https://hooks.slack.com/...
API_KEY=long-secret-key
HITL_DEFAULT_MODE=plan_only
MAX_STEP_RETRIES=3
KERNEL_EXECUTION_TIMEOUT=60
CHECKPOINT_TTL_HOURS=168        # 7 days
```

### Testing profile

```env
ENVIRONMENT=testing
CHECKPOINT_BACKEND=memory       # No disk persistence during tests
LLM_PROVIDER=mock               # Mock provider for tests (no real network calls)
LOG_LEVEL=WARNING
LOG_FORMAT=text
LANGCHAIN_TRACING_V2=false
MAX_STEP_RETRIES=1
KERNEL_EXECUTION_TIMEOUT=10
```

---

## Notes on Secure API Key Handling

API keys must never be committed to the repository. `.env` is gitignored. In production, environment variables are injected by the deployment system (e.g., container env vars, Kubernetes secrets manager, AWS Secrets Manager).

The repository's `.env.example` contains all variables with placeholder values (never real secrets) and serves as documentation of required variables.

---

## Startup Configuration Validation

On startup, before initializing any component, the system validates configuration to verify:

- Required API keys are present for the selected LLM provider
- The SQLite file is in a writable directory (or can be created)
- Upload and output directories exist and are writable
- The IPython kernel is installed and reachable
- Threshold values are consistent (e.g., warning threshold must be lower than error threshold)

In `development` and `testing`, the application can start even if provider API keys are not set. In that case, any operation that requires an LLM call will fail until the missing credentials are configured. In `production`, missing required credentials cause startup to fail.

If validation fails, the system refuses to start with a clear error message indicating exactly what is missing or incorrect.
