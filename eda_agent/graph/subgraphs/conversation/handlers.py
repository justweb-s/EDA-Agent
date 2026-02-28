"""Conversation handlers."""

from __future__ import annotations

from typing import Any


def data_question_handler(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def plan_modifier(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def general_responder(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
