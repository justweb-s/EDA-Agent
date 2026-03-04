from __future__ import annotations

import pytest

from eda_agent.llm_factory import reset_mock_llm_index


@pytest.fixture(autouse=True)
def _force_mock_llm_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    monkeypatch.setenv("LLM_MODEL", "mock-model")
    reset_mock_llm_index()
