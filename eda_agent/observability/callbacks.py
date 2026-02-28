"""LangChain callback handlers.

Implementations will be added once the LLM pipeline is wired in.
"""

from __future__ import annotations

from typing import Any


def build_callbacks(*args: Any, **kwargs: Any) -> list[Any]:
    return []
