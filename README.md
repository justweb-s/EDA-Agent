# EDA Agent

EDA Agent is a **single-user**, LangGraph-based multi-agent system that generates **complete, executed** exploratory data analysis (EDA) notebooks (`.ipynb`) from a CSV/Excel dataset.

## Documentation

The full project documentation lives in `docs/`:

- `docs/README.md`

## Quickstart (development)

1. Create a virtual environment (Python 3.11+).
2. Install the package:

```bash
pip install -e .[dev]
```

3. Copy environment variables:

```bash
cp .env.example .env
```

4. Run the API server:

```bash
python -m uvicorn eda_agent.api.main:app --reload
```

Then open:

- `GET http://127.0.0.1:8000/health`

## Project scope

- **Single-user only**
- Production-ready engineering practices (structured logging, configuration validation, testability)

