from __future__ import annotations

from pathlib import Path

import asyncio

import httpx
import pytest

from eda_agent.api.main import app


def _apply_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HITL_DEFAULT_MODE", "plan_only")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))


async def _create_session(client: httpx.AsyncClient, *, csv_bytes: bytes) -> str:
    resp = await client.post(
        "/sessions",
        files={"file": ("data.csv", csv_bytes, "text/csv")},
    )
    assert resp.status_code == 200
    return str(resp.json()["session_id"])


async def _wait_for_status(
    client: httpx.AsyncClient, *, session_id: str, status: str, timeout_s: float
) -> None:
    deadline = asyncio.get_event_loop().time() + timeout_s
    while asyncio.get_event_loop().time() < deadline:
        s = await client.get(f"/sessions/{session_id}")
        assert s.status_code == 200
        if s.json()["status"] == status:
            return
        await asyncio.sleep(0.05)
    raise AssertionError(f"Timed out waiting for status={status} session_id={session_id}")


async def _read_until_suspended(
    client: httpx.AsyncClient, *, session_id: str, timeout_s: float
) -> list[str]:
    events: list[str] = []

    async def _run() -> None:
        async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
            async for line in sse.aiter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    ev = line.split(":", 1)[1].strip()
                    events.append(ev)
                    if ev == "session_suspended":
                        return

    await asyncio.wait_for(_run(), timeout=timeout_s)
    return events


@pytest.mark.asyncio
async def test_two_sessions_suspend_without_interference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            s1, s2 = await asyncio.gather(
                _create_session(client, csv_bytes=b"a,b\n1,2\n3,4\n"),
                _create_session(client, csv_bytes=b"x,y\n5,6\n7,8\n"),
            )

            ev1_task = asyncio.create_task(
                _read_until_suspended(client, session_id=s1, timeout_s=15.0)
            )
            ev2_task = asyncio.create_task(
                _read_until_suspended(client, session_id=s2, timeout_s=15.0)
            )
            ev1, ev2 = await asyncio.gather(ev1_task, ev2_task)

            assert "hitl_interrupt" in ev1
            assert "session_suspended" in ev1
            assert "hitl_interrupt" in ev2
            assert "session_suspended" in ev2

            await _wait_for_status(client, session_id=s1, status="suspended", timeout_s=10.0)
            await _wait_for_status(client, session_id=s2, status="suspended", timeout_s=10.0)

            intr1 = await client.get(f"/sessions/{s1}/interrupt")
            intr2 = await client.get(f"/sessions/{s2}/interrupt")
            assert intr1.status_code == 200
            assert intr2.status_code == 200
            assert intr1.json()["session_id"] == s1
            assert intr2.json()["session_id"] == s2
