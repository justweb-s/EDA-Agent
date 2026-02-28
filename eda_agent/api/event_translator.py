"""Translate LangGraph stream events into SSE events.

The full translator will be implemented once the LangGraph graph is available.
"""

from __future__ import annotations


class EventTranslator:
    def __init__(self) -> None:
        pass

    def translate(self, event: object) -> tuple[str, dict]:
        raise NotImplementedError
