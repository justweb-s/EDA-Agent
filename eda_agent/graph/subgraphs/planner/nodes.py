"""Planner nodes."""

from __future__ import annotations

from typing import Any


def analyze_dataset(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def generate_plan(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError


def validate_plan(*args: Any, **kwargs: Any) -> Any:
    raise NotImplementedError
