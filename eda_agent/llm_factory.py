"""LLM factory.

This module is the only place where LLM providers are instantiated.
"""

from __future__ import annotations

from typing import Any, cast

from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import SecretStr

from eda_agent.config import EDAConfig


def get_llm(config: EDAConfig, *, callbacks: list[Any] | None = None) -> BaseChatModel:
    """Create and return a configured `BaseChatModel` instance.

    Provider-specific integrations are installed as optional dependencies.
    """

    provider = config.llm_provider.lower()

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.llm_max_retries,
            api_key=SecretStr(config.openai_api_key) if config.openai_api_key else None,
            callbacks=callbacks,
        )
        return cast(BaseChatModel, llm)

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        llm = ChatAnthropic(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.llm_max_retries,
            api_key=config.anthropic_api_key,
            callbacks=callbacks,
        )
        return cast(BaseChatModel, llm)

    if provider == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=config.llm_model,
            temperature=config.llm_temperature,
            max_retries=config.llm_max_retries,
            api_key=config.groq_api_key,
            callbacks=callbacks,
        )
        return cast(BaseChatModel, llm)

    if provider == "ollama":
        from langchain_ollama import ChatOllama

        llm = ChatOllama(
            model=config.llm_model,
            temperature=config.llm_temperature,
            base_url=config.ollama_base_url,
            callbacks=callbacks,
        )
        return cast(BaseChatModel, llm)

    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")
