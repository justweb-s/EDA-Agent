from __future__ import annotations

from pathlib import Path

import asyncio

import httpx
import pytest

from eda_agent.api.main import app


@pytest.mark.asyncio
async def test_hitl_interrupt_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HITL_DEFAULT_MODE", "plan_only")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))

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

            last_id = 0
            events: list[str] = []
            async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
                async for line in sse.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("id:"):
                        last_id = int(line.split(":", 1)[1].strip())
                    if line.startswith("event:"):
                        ev = line.split(":", 1)[1].strip()
                        events.append(ev)
                        if ev == "session_suspended":
                            break

            assert "hitl_interrupt" in events
            assert "session_suspended" in events

            deadline = asyncio.get_event_loop().time() + 10.0
            status_value = None
            while asyncio.get_event_loop().time() < deadline:
                s = await client.get(f"/sessions/{session_id}")
                assert s.status_code == 200
                status_value = s.json()["status"]
                if status_value in {"suspended", "completed", "failed"}:
                    break
                await asyncio.sleep(0.05)

            assert status_value == "suspended"

            intr = await client.get(f"/sessions/{session_id}/interrupt")
            assert intr.status_code == 200
            payload = intr.json()
            assert payload["session_id"] == session_id
            assert payload["interrupts"]
            assert payload["interrupts"][0]["value"]["type"] == "plan_approval"

            resume = await client.post(
                f"/sessions/{session_id}/resume",
                json={"resume": {"approved": True}},
            )
            assert resume.status_code == 200

            new_events: list[str] = []
            async with client.stream(
                "GET",
                f"/sessions/{session_id}/stream?from_event={last_id}",
            ) as sse2:
                async for line in sse2.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        ev = line.split(":", 1)[1].strip()
                        new_events.append(ev)
                        if ev == "session_completed":
                            break

            assert "session_completed" in new_events

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

            no_intr = await client.get(f"/sessions/{session_id}/interrupt")
            assert no_intr.status_code == 404


@pytest.mark.asyncio
async def test_resume_returns_409_when_not_suspended(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HITL_DEFAULT_MODE", "none")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))

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

            resume = await client.post(
                f"/sessions/{session_id}/resume",
                json={"resume": {"approved": True}},
            )
            assert resume.status_code == 409
