from __future__ import annotations

from typing import Any

import pytest

from eda_agent.exceptions import KernelExecutionError
from eda_agent.graph.subgraphs.eda_loop.executor import executor_node
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput


class _FakeRunResult:
    def __init__(self, *, ok: bool) -> None:
        self.execution_count = 1 if ok else None
        self.cell_outputs = [CellOutput(output_type="stream", text="ok")] if ok else []
        self.execution = ExecutionResult(
            stdout="ok" if ok else "",
            stderr="",
            success=ok,
            outputs=[],
        )


class _CrashThenOkKernel:
    def __init__(self, *, crash_message: str) -> None:
        self.crash_message = crash_message
        self.start_calls = 0
        self.shutdown_calls = 0
        self.execute_calls = 0

    def start(self, *, timeout_s: int = 30) -> None:
        self.start_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def execute(self, code: str, *, timeout_s: int, max_output_mb: float) -> Any:
        self.execute_calls += 1
        if self.execute_calls == 1:
            raise KernelExecutionError(self.crash_message)
        return _FakeRunResult(ok=True)


class _AlwaysKernelError:
    def __init__(self, *, message: str) -> None:
        self.message = message
        self.start_calls = 0
        self.shutdown_calls = 0
        self.execute_calls = 0

    def start(self, *, timeout_s: int = 30) -> None:
        self.start_calls += 1

    def shutdown(self) -> None:
        self.shutdown_calls += 1

    def execute(self, code: str, *, timeout_s: int, max_output_mb: float) -> Any:
        self.execute_calls += 1
        raise KernelExecutionError(self.message)


def test_executor_restarts_kernel_on_crash(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_RESTART_ON_CRASH", "true")
    kernel = _CrashThenOkKernel(crash_message="Kernel is not started")

    out = executor_node(
        {"generated_code": "print('hi')"},
        {"configurable": {"kernel": kernel}},
    )

    assert out["execution_count"] == 1
    assert out["execution_result"].success is True
    assert kernel.execute_calls == 2
    assert kernel.shutdown_calls == 1
    assert kernel.start_calls == 1


def test_executor_does_not_restart_on_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_RESTART_ON_CRASH", "true")
    kernel = _AlwaysKernelError(message="Kernel execution timed out")

    out = executor_node(
        {"generated_code": "print('hi')"},
        {"configurable": {"kernel": kernel}},
    )

    assert out["execution_result"].success is False
    assert out["execution_count"] is None
    assert kernel.execute_calls == 1
    assert kernel.shutdown_calls == 0
    assert kernel.start_calls == 0


def test_executor_does_not_restart_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KERNEL_RESTART_ON_CRASH", "false")
    kernel = _AlwaysKernelError(message="Kernel is not started")

    out = executor_node(
        {"generated_code": "print('hi')"},
        {"configurable": {"kernel": kernel}},
    )

    assert out["execution_result"].success is False
    assert kernel.execute_calls == 1
    assert kernel.shutdown_calls == 0
    assert kernel.start_calls == 0
