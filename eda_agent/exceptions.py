"""Domain-specific exceptions."""

from __future__ import annotations


class EDAAgentError(Exception):
    """Base exception for EDA Agent."""


class ConfigurationError(EDAAgentError):
    """Raised when configuration is invalid or incomplete."""


class DataIngestionError(EDAAgentError):
    """Raised when loading or profiling the dataset fails."""


class KernelExecutionError(EDAAgentError):
    """Raised when code execution in the IPython kernel fails."""


class PlanningError(EDAAgentError):
    """Raised when plan generation or validation fails."""
