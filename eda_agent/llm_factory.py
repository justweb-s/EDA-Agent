"""LLM factory.

This module is the only place where LLM providers are instantiated.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, cast

from langchain_core.language_models.chat_models import (
    BaseChatModel,
    SimpleChatModel,
)
from pydantic import SecretStr

from eda_agent.config import EDAConfig

_mock_llm_index = 0


def reset_mock_llm_index() -> None:
    global _mock_llm_index
    _mock_llm_index = 0


class MockChatModel(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "mock"

    def _call(
        self,
        messages: list[Any],
        stop: Any | None = None,
        run_manager: Any | None = None,
        **kwargs: Any,
    ) -> str:
        content = "\n".join(str(getattr(m, "content", m)) for m in messages)

        raw_override = os.getenv("MOCK_LLM_RESPONSE")
        if raw_override:
            return raw_override

        raw_list = os.getenv("MOCK_LLM_RESPONSES")
        if raw_list:
            try:
                items = json.loads(raw_list)
                if isinstance(items, list) and items:
                    global _mock_llm_index
                    idx = _mock_llm_index
                    if idx < len(items):
                        _mock_llm_index += 1
                        item = items[idx]
                        if isinstance(item, (dict, list)):
                            return json.dumps(item)
                        return str(item)
            except Exception:
                pass

        if "TASK: CODER" in content:
            cell: dict[str, Any] = {
                "code": "df.head()",
                "expected_output_description": "A small preview of the dataframe.",
            }
            return json.dumps(cell)

        if "TASK: OBSERVER" in content:
            if "EXECUTION_SUCCESS: false" in content or (
                "STDERR:" in content and "Traceback" in content
            ):
                verdict: dict[str, Any] = {
                    "verdict": "retry",
                    "findings_description": "The cell failed to execute.",
                    "error_analysis": "Execution error detected.",
                    "retry_hint": "Fix the error and retry the step.",
                }
                return json.dumps(verdict)

            verdict = {
                "verdict": "success",
                "findings_description": "Step executed successfully.",
                "key_statistics": {},
                "charts_produced": [],
                "anomalies_found": [],
            }
            return json.dumps(verdict)

        if "TASK: ANALYZE_DATASET" in content:
            return " ".join(
                [
                    "Dataset looks usable for EDA.",
                    "Proceed with basic quality checks and univariate summaries.",
                ]
            )

        if "TASK: GENERATE_PLAN" in content:
            if re.search(r"VALIDATION_ERRORS:\s*\[.+\]", content, flags=re.DOTALL):
                plan = {
                    "steps": [
                        {
                            "step_id": "data_quality",
                            "section": "Data quality",
                            "title": "Data quality checks",
                            "description": (
                                "Review missing values, duplicates, and detected issues."
                            ),
                            "analysis_type": "data_quality",
                            "target_columns": [],
                            "depends_on": [],
                            "is_mandatory": True,
                            "priority": 10,
                        },
                        {
                            "step_id": "univariate_numeric",
                            "section": "Univariate",
                            "title": "Numeric distributions",
                            "description": (
                                "Inspect distributions and summary statistics for numeric columns."
                            ),
                            "analysis_type": "univariate",
                            "target_columns": [],
                            "depends_on": ["data_quality"],
                            "is_mandatory": True,
                            "priority": 8,
                        },
                        {
                            "step_id": "univariate_categorical",
                            "section": "Univariate",
                            "title": "Categorical distributions",
                            "description": (
                                "Inspect value counts and rare categories for categorical columns."
                            ),
                            "analysis_type": "univariate",
                            "target_columns": [],
                            "depends_on": ["data_quality"],
                            "is_mandatory": True,
                            "priority": 7,
                        },
                    ]
                }
                return json.dumps(plan)

            invalid_plan = {
                "steps": [
                    {
                        "step_id": "data_quality",
                        "section": "Data quality",
                        "title": "Data quality checks",
                        "description": ("Review missing values, duplicates, and detected issues."),
                        "analysis_type": "data_quality",
                        "target_columns": [],
                        "depends_on": [],
                        "is_mandatory": True,
                        "priority": 10,
                    },
                    {
                        "step_id": "data_quality",
                        "section": "Univariate",
                        "title": "Numeric distributions",
                        "description": (
                            "Inspect distributions and summary statistics for numeric columns."
                        ),
                        "analysis_type": "univariate",
                        "target_columns": [],
                        "depends_on": ["data_quality"],
                        "is_mandatory": True,
                        "priority": 8,
                    },
                ]
            }
            return json.dumps(invalid_plan)

        default_plan = {
            "steps": [
                {
                    "step_id": "data_quality",
                    "section": "Data quality",
                    "title": "Data quality checks",
                    "description": ("Review missing values, duplicates, and detected issues."),
                    "analysis_type": "data_quality",
                    "target_columns": [],
                    "depends_on": [],
                    "is_mandatory": True,
                    "priority": 10,
                }
            ]
        }
        return json.dumps(default_plan)


def get_llm(config: EDAConfig, *, callbacks: Any | None = None) -> BaseChatModel:
    """Create and return a configured `BaseChatModel` instance.

    Provider-specific integrations are installed as optional dependencies.
    """

    provider = config.llm_provider.lower()

    if provider == "mock":
        return cast(BaseChatModel, MockChatModel())

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return cast(
            BaseChatModel,
            ChatOpenAI(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_retries=config.llm_max_retries,
                api_key=SecretStr(config.openai_api_key) if config.openai_api_key else None,
                callbacks=callbacks,
            ),
        )

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return cast(
            BaseChatModel,
            ChatAnthropic(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_retries=config.llm_max_retries,
                api_key=config.anthropic_api_key,
                callbacks=callbacks,
            ),
        )

    if provider == "groq":
        from langchain_groq import ChatGroq

        return cast(
            BaseChatModel,
            ChatGroq(
                model=config.llm_model,
                temperature=config.llm_temperature,
                max_retries=config.llm_max_retries,
                api_key=config.groq_api_key,
                callbacks=callbacks,
            ),
        )

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        return cast(
            BaseChatModel,
            ChatOllama(
                model=config.llm_model,
                temperature=config.llm_temperature,
                base_url=config.ollama_base_url,
                callbacks=callbacks,
            ),
        )

    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
