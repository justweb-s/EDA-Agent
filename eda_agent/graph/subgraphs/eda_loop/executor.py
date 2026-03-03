"""EXECUTOR node (runs code in the kernel)."""

from __future__ import annotations

from typing import Any, cast

from langchain_core.runnables import RunnableConfig

from eda_agent.config import EDAConfig
from eda_agent.graph.subgraphs.eda_loop.state import EDALoopState
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput
from eda_agent.tools.kernel import create_kernel


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
        run_result = kernel.execute(
            generated_code,
            timeout_s=cfg.kernel_execution_timeout,
            max_output_mb=cfg.kernel_max_output_size_mb,
        )
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
