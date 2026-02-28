"""Context builder.

Builds agent-specific prompt variables from global state.
"""

from __future__ import annotations

from typing import Any


def build_agent_context(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
