from __future__ import annotations

from pathlib import Path

import asyncio

import httpx
import pytest

from eda_agent.api.main import app


def _apply_test_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("HITL_DEFAULT_MODE", "none")
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "outputs"))
    monkeypatch.setenv("UPLOAD_DIR", str(tmp_path / "uploads"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "checkpoints.db"))


@pytest.mark.asyncio
async def test_create_session_requires_file_field(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sessions",
                files={},
            )

    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_create_session_rejects_unsupported_extension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/sessions",
                files={"file": ("data.txt", b"hello", "text/plain")},
            )

    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_download_and_cells_404_for_unknown_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            nb = await client.get("/sessions/does-not-exist/notebook")
            cells = await client.get("/sessions/does-not-exist/notebook/cells")
            stream = await client.get("/sessions/does-not-exist/stream")

    assert nb.status_code == 404
    assert cells.status_code == 404
    assert stream.status_code == 404


@pytest.mark.asyncio
async def test_hitl_endpoints_error_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            missing_resume = await client.post(
                "/sessions/does-not-exist/resume",
                json={"resume": {"approved": True}},
            )
            assert missing_resume.status_code == 404

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

            no_intr = await client.get(f"/sessions/{session_id}/interrupt")
            assert no_intr.status_code == 404


@pytest.mark.asyncio
async def test_list_sessions_and_delete_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            empty = await client.get("/sessions")
            assert empty.status_code == 200
            payload = empty.json()
            assert "sessions" in payload
            assert isinstance(payload["sessions"], list)

            missing_delete = await client.delete("/sessions/does-not-exist")
            assert missing_delete.status_code == 404


@pytest.mark.asyncio
async def test_resume_schema_validation_422(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _apply_test_env(monkeypatch, tmp_path)

    async with app.router.lifespan_context(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/sessions/does-not-exist/resume", json={})

    assert resp.status_code == 422
