"""Logging configuration utilities."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from typing import Any

from eda_agent.config import EDAConfig


class JsonFormatter(logging.Formatter):
    """Minimal JSON log formatter with stable keys."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include known structured fields when present.
        for key in (
            "session_id",
            "step_id",
            "execution_time_ms",
            "output_types",
            "code_lines",
        ):
            if hasattr(record, key):
                payload[key] = getattr(record, key)

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, ensure_ascii=False)


def configure_logging(config: EDAConfig) -> None:
    """Configure application-wide logging."""

    level = getattr(logging, config.log_level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    handler = logging.StreamHandler()
    if config.log_format == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s - %(message)s"))

    root.handlers = [handler]
