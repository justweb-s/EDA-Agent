from __future__ import annotations

import asyncio
from pathlib import Path

import httpx
import pytest

from eda_agent.api.main import app


@pytest.mark.asyncio
async def test_minimal_run_produces_cells_and_notebook() -> None:
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

            cells_resp = await client.get(f"/sessions/{session_id}/notebook/cells")
            assert cells_resp.status_code == 200
            cells = cells_resp.json()
            assert isinstance(cells, list)
            assert len(cells) >= 2

            nb_resp = await client.get(f"/sessions/{session_id}/notebook")
            assert nb_resp.status_code == 200
            assert nb_resp.headers["content-type"].startswith("application/x-ipynb+json")
            assert len(nb_resp.content) > 100

            notebook_path = (
                Path("./outputs") / "notebooks" / f"eda-agent-{session_id}.ipynb"
            ).resolve()
            assert notebook_path.exists()
            assert notebook_path.stat().st_size > 100


@pytest.mark.asyncio
async def test_sse_stream_emits_completion() -> None:
    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            csv_bytes = b"x\n1\n2\n3\n"
            resp = await client.post(
                "/sessions",
                files={"file": ("data.csv", csv_bytes, "text/csv")},
            )
            assert resp.status_code == 200
            session_id = resp.json()["session_id"]

            async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
                assert sse.status_code == 200
                assert sse.headers["content-type"].startswith("text/event-stream")

                events = []
                async for line in sse.aiter_lines():
                    if not line:
                        continue
                    if line.startswith("event:"):
                        events.append(line.split(":", 1)[1].strip())

            assert "session_completed" in events
            assert "cell_added" in events
