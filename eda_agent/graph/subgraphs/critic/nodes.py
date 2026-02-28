"""Critic nodes."""

from __future__ import annotations

from typing import Any


def read_section(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def evaluate_section(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
