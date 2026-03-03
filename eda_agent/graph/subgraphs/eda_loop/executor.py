"""EXECUTOR node (runs code in the kernel)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from eda_agent.config import EDAConfig
from eda_agent.exceptions import KernelExecutionError
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput
from eda_agent.tools.kernel import create_kernel


def _is_kernel_crash_error(err: KernelExecutionError) -> bool:
    msg = str(err).lower()
    crash_markers = (
        "kernel is not started",
        "did not become ready",
        "parent appears to have exited",
        "channels",
        "iopub",
    )
    return any(marker in msg for marker in crash_markers)


def executor_node(*args: Any, **kwargs: Any) -> Any:
    state = cast(EDALoopState, args[0] if args else kwargs.get("state"))
    config = cast(
        RunnableConfig | None,
        (args[1] if len(args) > 1 else kwargs.get("config")),
    )

    generated_code = str(state.get("generated_code") or "")
    cfg = EDAConfig()

    kernel: Any | None = None
    should_shutdown = False
    if config is not None:
        configurable = config.get("configurable", {}) or {}
        kernel = configurable.get("kernel")

    if kernel is None:
        kernel = create_kernel()
        should_shutdown = True
        kernel.start(timeout_s=30)

    try:
        def _run() -> Any:
            return kernel.execute(
                generated_code,
                timeout_s=cfg.kernel_execution_timeout,
                max_output_mb=cfg.kernel_max_output_size_mb,
            )

        try:
            run_result = _run()
        except KernelExecutionError as e:
            if cfg.kernel_restart_on_crash and _is_kernel_crash_error(e):
                try:
                    kernel.shutdown()
                except Exception:
                    pass
                try:
                    kernel.start(timeout_s=30)
                    run_result = _run()
                except Exception as retry_err:  # noqa: BLE001
                    return {
                        "execution_count": None,
                        "cell_outputs": [
                            CellOutput(output_type="error", text=str(retry_err))
                        ],
                        "execution_result": ExecutionResult(
                            stdout="",
                            stderr=str(retry_err),
                            success=False,
                            outputs=[],
                        ),
                    }
            else:
                return {
                    "execution_count": None,
                    "cell_outputs": [CellOutput(output_type="error", text=str(e))],
                    "execution_result": ExecutionResult(
                        stdout="",
                        stderr=str(e),
                        success=False,
                        outputs=[],
                    ),
                }

        return {
            "execution_count": run_result.execution_count,
            "cell_outputs": run_result.cell_outputs,
            "execution_result": run_result.execution,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "execution_count": None,
            "cell_outputs": [CellOutput(output_type="error", text=str(e))],
            "execution_result": ExecutionResult(
                stdout="", stderr=str(e), success=False, outputs=[]
            ),
        }
    finally:
        if should_shutdown:
            try:
                kernel.shutdown()
            except Exception:
                pass
