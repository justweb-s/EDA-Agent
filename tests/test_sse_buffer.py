from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from eda_agent.api.main import app


def _apply_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HITL_DEFAULT_MODE", "none")
    monkeypatch.setenv("SSE_BUFFER_SIZE", "1")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))


@pytest.mark.asyncio
async def test_small_sse_buffer_still_replays_terminal_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            csv_bytes = b"a,b\n1,2\n3,4\n"
            resp = await client.post(
                "/sessions",
                files={"file": ("data.csv", csv_bytes, "text/csv")},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

            deadline = asyncio.get_event_loop().time() + 15.0
            status_value = None
            while asyncio.get_event_loop().time() < deadline:
                s = await client.get(f"/sessions/{session_id}")
                assert s.status_code == 200
                status_value = s.json()["status"]
                if status_value in {"completed", "failed"}:
                    break
                await asyncio.sleep(0.05)

            assert status_value == "completed"

            events: list[str] = []

            async def _read() -> None:
                async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
                    async for line in sse.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("event:"):
                            events.append(line.split(":", 1)[1].strip())

            await asyncio.wait_for(_read(), timeout=5.0)
            assert events[-1] == "session_completed"
