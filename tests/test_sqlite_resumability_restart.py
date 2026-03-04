from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import pytest

from eda_agent.api.main import app
from eda_agent.models.execution import ExecutionResult
from eda_agent.models.notebook import CellOutput


class _FakeRunResult:
    def __init__(self) -> None:
        self.execution_count = 1
        self.cell_outputs = [CellOutput(output_type="stream", text="ok")]
        self.execution = ExecutionResult(stdout="ok", stderr="", success=True, outputs=[])


class _FakeKernel:
    def start(self, *, timeout_s: int = 30) -> None:  # noqa: ARG002
        return

    def shutdown(self) -> None:
        return

    def execute(self, code: str, *, timeout_s: int, max_output_mb: float) -> Any:  # noqa: ARG002
        return _FakeRunResult()


def _apply_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("HITL_DEFAULT_MODE", "plan_only")
    monkeypatch.setenv("SSE_BUFFER_SIZE", "250")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))


async def _wait_status(
    client: httpx.AsyncClient, *, session_id: str, want: set[str], timeout_s: float
) -> str:
    deadline = asyncio.get_event_loop().time() + timeout_s
    status_value = None
    while asyncio.get_event_loop().time() < deadline:
        s = await client.get(f"/sessions/{session_id}")
        assert s.status_code == 200
        status_value = str(s.json()["status"])
        if status_value in want:
            return status_value
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for status in {want}. Last={status_value}")


@pytest.mark.asyncio
async def test_sqlite_resume_survives_process_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    from eda_agent.api import session_runner as session_runner_module

    monkeypatch.setattr(session_runner_module, "create_kernel", lambda: _FakeKernel())

    csv_bytes = b"a,b\n1,2\n3,4\n"

    session_id: str

    # Phase 1: create session, run until HITL suspend.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sessions",
                files={"file": ("data.csv", csv_bytes, "text/csv")},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

            events1: list[str] = []
            async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
                async for line in sse.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        ev = line.split(":", 1)[1].strip()
                        events1.append(ev)
                        if ev == "session_suspended":
                            break

            assert "plan_generated" in events1
            assert "hitl_interrupt" in events1
            assert "session_suspended" in events1

            status_value = await _wait_status(
                client, session_id=session_id, want={"suspended", "failed"}, timeout_s=10.0
            )
            assert status_value == "suspended"

    # Phase 2: "restart" the server (new lifespan -> new state objects) and resume.
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            status_value = await _wait_status(
                client, session_id=session_id, want={"suspended", "failed"}, timeout_s=5.0
            )
            assert status_value == "suspended"

            intr = await client.get(f"/sessions/{session_id}/interrupt")
            assert intr.status_code == 200
            payload = intr.json()
            assert payload["session_id"] == session_id
            assert payload["interrupts"]
            assert payload["interrupts"][0]["value"]["type"] == "plan_approval"

            resume = await client.post(
                f"/sessions/{session_id}/resume",
                json={"resume": {"approved": True, "received_at": datetime.now(UTC).isoformat()}},
            )
            assert resume.status_code == 200

            events2: list[str] = []
            plan_cell_count = 0
            current_event: str | None = None

            async with client.stream("GET", f"/sessions/{session_id}/stream") as sse2:
                async for line in sse2.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        events2.append(current_event)
                        continue

                    if current_event != "cell_added":
                        continue

                    if not line.startswith("data:"):
                        continue

                    data = json.loads(line.split(":", 1)[1].strip())
                    cell = data.get("cell") or {}
                    if "## Plan" in str(cell.get("source") or ""):
                        plan_cell_count += 1

            assert "plan_generated" not in events2
            assert "hitl_interrupt" not in events2
            assert "session_completed" in events2
            assert plan_cell_count == 1

            status_value = await _wait_status(
                client, session_id=session_id, want={"completed", "failed"}, timeout_s=20.0
            )
            assert status_value == "completed"

            nb_resp = await client.get(f"/sessions/{session_id}/notebook")
            assert nb_resp.status_code == 200
            assert nb_resp.headers["content-type"].startswith("application/x-ipynb+json")
            assert len(nb_resp.content) > 100

            notebook_path = (
                tmp_path / "outputs" / "notebooks" / f"eda-agent-{session_id}.ipynb"
            ).resolve()
            assert notebook_path.exists()
            assert notebook_path.stat().st_size > 100

            no_intr = await client.get(f"/sessions/{session_id}/interrupt")
            assert no_intr.status_code == 404
