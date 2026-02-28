"""IPython kernel wrapper.

Will be implemented in a later milestone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, cast

from jupyter_client import KernelManager
from jupyter_client.blocking import BlockingKernelClient

from eda_agent.exceptions import KernelExecutionError
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput


@dataclass(frozen=True)
class KernelRunResult:
    execution_count: int | None
    execution: ExecutionResult
    cell_outputs: list[CellOutput]


class IPythonKernel:
    def __init__(self, *, kernel_name: str = "python3") -> None:
        self._kernel_name = kernel_name
        self._km: KernelManager | None = None
        self._client: BlockingKernelClient | None = None

    def start(self, *, timeout_s: int = 30) -> None:
        if self._km is not None:
            return

        km = KernelManager(kernel_name=self._kernel_name)
        km.start_kernel()
        client = km.client()
        client.start_channels()

        try:
            client.wait_for_ready(timeout=timeout_s)
        except Exception as e:  # noqa: BLE001
            try:
                client.stop_channels()
            finally:
                km.shutdown_kernel(now=True)
                km.cleanup_resources()
            raise KernelExecutionError(f"Kernel did not become ready: {e}") from e

        self._km = km
        self._client = client

    def shutdown(self) -> None:
        client = self._client
        km = self._km

        self._client = None
        self._km = None

        if client is not None:
            try:
                client.stop_channels()
            except Exception:
                pass

        if km is not None:
            try:
                km.shutdown_kernel(now=True)
            except Exception:
                pass
            try:
                km.cleanup_resources()
            except Exception:
                pass

    def execute(
        self,
        code: str,
        *,
        timeout_s: int,
        max_output_mb: float = 5.0,
        store_history: bool = True,
    ) -> KernelRunResult:
        if self._client is None:
            raise KernelExecutionError("Kernel is not started")

        client = self._client
        msg_id = cast(str, client.execute(code, store_history=store_history))

        execution_count: int | None = None
        stdout_parts: list[str] = []
        stderr_parts: list[str] = []
        cell_outputs: list[CellOutput] = []
        raw_outputs: list[dict[str, Any]] = []
        success = True

        deadline = time.monotonic() + timeout_s
        max_bytes = int(max_output_mb * 1024 * 1024)

        try:
            while time.monotonic() < deadline:
                remaining = max(0.0, deadline - time.monotonic())
                msg = client.get_iopub_msg(timeout=remaining)

                parent = cast(dict[str, Any], msg.get("parent_header") or {})
                if parent.get("msg_id") != msg_id:
                    continue

                msg_type = cast(str, cast(dict[str, Any], msg.get("header") or {}).get("msg_type"))
                content = cast(dict[str, Any], msg.get("content") or {})

                if msg_type == "status" and content.get("execution_state") == "idle":
                    break

                if msg_type == "stream":
                    text = str(content.get("text") or "")
                    name = str(content.get("name") or "stdout")
                    if name == "stderr":
                        stderr_parts.append(text)
                    else:
                        stdout_parts.append(text)
                    cell_outputs.append(CellOutput(output_type="stream", text=text))
                    raw_outputs.append({"output_type": "stream", "name": name, "text": text})

                elif msg_type in {"display_data", "execute_result"}:
                    data = _coerce_output_data(cast(dict[str, Any], content.get("data") or {}))
                    cell_outputs.append(CellOutput(output_type=msg_type, data=data))
                    raw_outputs.append({"output_type": msg_type, "data": data})

                elif msg_type == "error":
                    success = False
                    tb = "\n".join([str(line) for line in (content.get("traceback") or [])])
                    text = tb or str(content.get("evalue") or "")
                    cell_outputs.append(CellOutput(output_type="error", text=text))
                    raw_outputs.append(
                        {
                            "output_type": "error",
                            "ename": str(content.get("ename") or "Error"),
                            "evalue": str(content.get("evalue") or ""),
                            "traceback": [str(line) for line in (content.get("traceback") or [])],
                        }
                    )

                if _estimate_output_size(stdout_parts, stderr_parts, raw_outputs) > max_bytes:
                    raise KernelExecutionError("Kernel output exceeded max size")

            else:
                raise KernelExecutionError("Kernel execution timed out")

            remaining = max(0.0, deadline - time.monotonic())
            reply = client.get_shell_msg(timeout=remaining)
            reply_content = cast(dict[str, Any], reply.get("content") or {})
            execution_count = cast(int | None, reply_content.get("execution_count"))

        except Exception as e:  # noqa: BLE001
            raise KernelExecutionError(str(e)) from e

        execution = ExecutionResult(
            stdout="".join(stdout_parts),
            stderr="".join(stderr_parts),
            success=success,
            outputs=raw_outputs,
        )
        return KernelRunResult(
            execution_count=execution_count,
            execution=execution,
            cell_outputs=cell_outputs,
        )


def create_kernel(*args: Any, **kwargs: Any) -> IPythonKernel:
    return IPythonKernel(*args, **kwargs)


def _coerce_output_data(data: dict[str, Any]) -> dict[str, str]:
    coerced: dict[str, str] = {}
    for k, v in data.items():
        if v is None:
            continue
        if isinstance(v, str):
            coerced[str(k)] = v
        else:
            coerced[str(k)] = str(v)
    return coerced


def _estimate_output_size(
    stdout_parts: list[str],
    stderr_parts: list[str],
    outputs: list[dict[str, Any]],
) -> int:
    size = sum(len(s.encode("utf-8")) for s in stdout_parts)
    size += sum(len(s.encode("utf-8")) for s in stderr_parts)
    size += sum(len(str(o).encode("utf-8")) for o in outputs)
    return size
