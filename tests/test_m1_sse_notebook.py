from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from eda_agent.api.main import app


@pytest.mark.asyncio
async def test_minimal_run_produces_cells_and_notebook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

            assert any(
                (c.get("cell_type") == "markdown") and ("## Plan" in str(c.get("source", "")))
                for c in cells
            )

            assert any(
                (c.get("cell_type") == "markdown")
                and (
                    c.get("step_id")
                    in {"data_quality", "univariate_numeric", "univariate_categorical"}
                )
                for c in cells
            )

            nb_resp = await client.get(f"/sessions/{session_id}/notebook")
            assert nb_resp.status_code == 200
            assert nb_resp.headers["content-type"].startswith("application/x-ipynb+json")
            assert len(nb_resp.content) > 100

            notebook_path = (
                (tmp_path / "outputs") / "notebooks" / f"eda-agent-{session_id}.ipynb"
            ).resolve()
            assert notebook_path.exists()
            assert notebook_path.stat().st_size > 100


@pytest.mark.asyncio
async def test_sse_stream_emits_completion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))

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
            assert "plan_generated" in events


@pytest.mark.asyncio
async def test_sse_stream_does_not_duplicate_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

            seen: set[tuple[str, str, str]] = set()
            duplicates: list[tuple[str, str, str]] = []

            async with client.stream("GET", f"/sessions/{session_id}/stream") as sse:
                assert sse.status_code == 200

                current_event: str | None = None
                async for line in sse.aiter_lines():
                    if not line:
                        continue

                    if line.startswith("event:"):
                        current_event = line.split(":", 1)[1].strip()
                        continue

                    if current_event != "cell_added":
                        continue

                    if not line.startswith("data:"):
                        continue

                    payload = line.split(":", 1)[1].strip()
                    data = json.loads(payload)
                    cell = data.get("cell") or {}
                    key = (
                        str(cell.get("cell_type") or ""),
                        str(cell.get("source") or ""),
                        str(cell.get("step_id") or ""),
                    )
                    if key in seen:
                        duplicates.append(key)
                    else:
                        seen.add(key)

            assert not duplicates


@pytest.mark.asyncio
async def test_download_notebook_conflict_then_ok(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

            early = await client.get(f"/sessions/{session_id}/notebook")
            assert early.status_code == 409

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

            ok = await client.get(f"/sessions/{session_id}/notebook")
            assert ok.status_code == 200
            assert ok.headers["content-type"].startswith("application/x-ipynb+json")
            assert len(ok.content) > 100


@pytest.mark.asyncio
async def test_delete_session_removes_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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

            deleted = await client.delete(f"/sessions/{session_id}")
            assert deleted.status_code == 200
            assert deleted.json()["status"] == "deleted"

            missing = await client.get(f"/sessions/{session_id}")
            assert missing.status_code == 404
