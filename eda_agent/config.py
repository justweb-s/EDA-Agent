"""Application configuration.

All configuration is loaded via environment variables (and optionally a .env file).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


def _parse_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v) for v in value]
    if isinstance(value, tuple):
        return [str(v) for v in value]
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return []
        if s.startswith("["):
            return [str(v) for v in json.loads(s)]
        return [part.strip() for part in s.split(",") if part.strip()]
    return [str(value)]


class EDAConfig(BaseSettings):
    """Global configuration model for the EDA Agent."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore", populate_by_name=True)

    environment: Literal["development", "production", "testing"] = Field(
        default="development", validation_alias="ENVIRONMENT"
    )

    # LLM
    llm_provider: str = Field(default="openai", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="gpt-4o", validation_alias="LLM_MODEL")
    llm_temperature: float = Field(default=0.1, validation_alias="LLM_TEMPERATURE")
    llm_max_retries: int = Field(default=3, validation_alias="LLM_MAX_RETRIES")

    openai_api_key: str | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    groq_api_key: str | None = Field(default=None, validation_alias="GROQ_API_KEY")
    ollama_base_url: str = Field(
        default="http://localhost:11434",
        validation_alias="OLLAMA_BASE_URL",
    )

    # Checkpointing
    checkpoint_backend: Literal["memory", "sqlite"] = Field(
        default="sqlite", validation_alias="CHECKPOINT_BACKEND"
    )
    sqlite_path: Path = Field(default=Path("./checkpoints.db"), validation_alias="SQLITE_PATH")
    checkpoint_ttl_hours: int = Field(default=72, validation_alias="CHECKPOINT_TTL_HOURS")

    # Kernel execution
    kernel_execution_timeout: int = Field(default=60, validation_alias="KERNEL_EXECUTION_TIMEOUT")
    kernel_heavy_timeout: int = Field(default=300, validation_alias="KERNEL_HEAVY_TIMEOUT")
    kernel_restart_on_crash: bool = Field(default=True, validation_alias="KERNEL_RESTART_ON_CRASH")
    kernel_max_output_size_mb: float = Field(
        default=5.0,
        validation_alias="KERNEL_MAX_OUTPUT_SIZE_MB",
    )

    # Analysis parameters
    max_step_retries: int = Field(default=3, validation_alias="MAX_STEP_RETRIES")
    max_critic_iterations: int = Field(default=2, validation_alias="MAX_CRITIC_ITERATIONS")
    mandatory_sections: list[str] = Field(
        default_factory=lambda: ["data_quality", "overview"],
        validation_alias="MANDATORY_SECTIONS",
    )
    null_warning_threshold: float = Field(default=0.05, validation_alias="NULL_WARNING_THRESHOLD")
    null_error_threshold: float = Field(default=0.30, validation_alias="NULL_ERROR_THRESHOLD")
    outlier_sigma_threshold: float = Field(default=3.0, validation_alias="OUTLIER_SIGMA_THRESHOLD")
    high_cardinality_threshold: int = Field(
        default=20,
        validation_alias="HIGH_CARDINALITY_THRESHOLD",
    )

    # Memory management
    context_window_warning_pct: float = Field(
        default=0.80,
        validation_alias="CONTEXT_WINDOW_WARNING_PCT",
    )
    summarization_keep_last_n: int = Field(default=10, validation_alias="SUMMARIZATION_KEEP_LAST_N")
    execution_history_max_section_detail: int = Field(
        default=5, validation_alias="EXECUTION_HISTORY_MAX_SECTION_DETAIL"
    )

    # HITL
    hitl_default_mode: Literal["none", "plan_only", "full", "custom"] = Field(
        default="none", validation_alias="HITL_DEFAULT_MODE"
    )
    hitl_timeout_hours: int = Field(default=24, validation_alias="HITL_TIMEOUT_HOURS")
    hitl_touchpoints: list[str] = Field(
        default_factory=lambda: ["plan_approval", "critic_review"],
        validation_alias="HITL_TOUCHPOINTS",
    )

    # Observability
    langchain_tracing_v2: bool = Field(default=False, validation_alias="LANGCHAIN_TRACING_V2")
    langchain_api_key: str | None = Field(default=None, validation_alias="LANGCHAIN_API_KEY")
    langchain_project: str = Field(default="eda-agent", validation_alias="LANGCHAIN_PROJECT")

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    log_format: Literal["json", "text"] = Field(default="json", validation_alias="LOG_FORMAT")

    alert_channel: Literal["log", "email", "slack", "webhook"] = Field(
        default="log", validation_alias="ALERT_CHANNEL"
    )
    alert_slack_webhook_url: str | None = Field(
        default=None,
        validation_alias="ALERT_SLACK_WEBHOOK_URL",
    )
    alert_email_recipients: list[str] = Field(
        default_factory=list, validation_alias="ALERT_EMAIL_RECIPIENTS"
    )

    # API
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    api_key: str | None = Field(default=None, validation_alias="API_KEY")

    upload_dir: Path = Field(default=Path("./uploads"), validation_alias="UPLOAD_DIR")
    output_dir: Path = Field(default=Path("./outputs"), validation_alias="OUTPUT_DIR")
    max_upload_size_mb: float = Field(default=100.0, validation_alias="MAX_UPLOAD_SIZE_MB")
    sse_buffer_size: int = Field(default=100, validation_alias="SSE_BUFFER_SIZE")

    @field_validator(
        "mandatory_sections",
        "hitl_touchpoints",
        "alert_email_recipients",
        mode="before",
    )
    @classmethod
    def _validate_list_fields(cls, v: Any) -> list[str]:
        return _parse_list(v)

    @field_validator("null_error_threshold")
    @classmethod
    def _validate_thresholds(cls, v: float, info: Any) -> float:
        warning = info.data.get("null_warning_threshold")
        if warning is not None and warning >= v:
            raise ValueError("NULL_WARNING_THRESHOLD must be lower than NULL_ERROR_THRESHOLD")
        return v


def validate_config(config: EDAConfig) -> None:
    """Validate configuration and raise a ValueError with actionable messages."""

    if config.environment == "production":
        provider = config.llm_provider.lower()
        if provider == "openai" and not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        if provider == "anthropic" and not config.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY is required when LLM_PROVIDER=anthropic")
        if provider == "groq" and not config.groq_api_key:
            raise ValueError("GROQ_API_KEY is required when LLM_PROVIDER=groq")
        if not config.api_key:
            raise ValueError("API_KEY is required when ENVIRONMENT=production")

    if config.checkpoint_backend == "sqlite":
        parent = config.sqlite_path.expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    config.upload_dir.mkdir(parents=True, exist_ok=True)
    config.output_dir.mkdir(parents=True, exist_ok=True)
