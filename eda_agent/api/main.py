"""FastAPI application entry point."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI

from eda_agent.config import EDAConfig, validate_config
from eda_agent.observability.logging_config import configure_logging

from .dependencies import require_api_key
from .routes.download import router as download_router
from .routes.health import router as health_router
from .routes.hitl import router as hitl_router
from .routes.metrics import router as metrics_router
from .routes.sessions import router as sessions_router
from .routes.stream import router as stream_router
from .session_store import SessionStore
from .sse_broker import SSEBroker


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    config = EDAConfig()
    validate_config(config)
    configure_logging(config)

    app.state.config = config
    app.state.session_store = SessionStore(output_dir=config.output_dir)
    app.state.sse_broker = SSEBroker(buffer_size=config.sse_buffer_size)
    yield


app = FastAPI(
    title="EDA Agent",
    version="0.1.0",
    lifespan=lifespan,
    dependencies=[Depends(require_api_key)],
)

app.include_router(health_router)
app.include_router(metrics_router)
app.include_router(sessions_router)
app.include_router(stream_router)
app.include_router(hitl_router)
app.include_router(download_router)
