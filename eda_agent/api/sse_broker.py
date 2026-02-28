from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator
from dataclasses import dataclass


@dataclass(frozen=True)
class SSEMessage:
    message_id: int
    event: str
    data: dict


class SessionChannel:
    def __init__(self, *, buffer_size: int) -> None:
        self._buffer: deque[SSEMessage] = deque(maxlen=buffer_size)
        self._next_id = 1
        self._closed = False
        self._cond = asyncio.Condition()

    async def publish(self, event: str, data: dict) -> SSEMessage:
        async with self._cond:
            msg = SSEMessage(message_id=self._next_id, event=event, data=data)
            self._next_id += 1
            self._buffer.append(msg)
            self._cond.notify_all()
            return msg

    async def close(self) -> None:
        async with self._cond:
            self._closed = True
            self._cond.notify_all()

    async def subscribe(self, *, from_event: int = 0) -> AsyncIterator[SSEMessage]:
        last_sent = from_event

        while True:
            async with self._cond:
                pending = [m for m in self._buffer if m.message_id > last_sent]
                closed = self._closed
                if not pending and not closed:
                    await self._cond.wait()
                    continue

            for msg in pending:
                last_sent = msg.message_id
                yield msg

            if closed and not pending:
                return


class SSEBroker:
    def __init__(self, *, buffer_size: int) -> None:
        self._buffer_size = buffer_size
        self._channels: dict[str, SessionChannel] = {}
        self._lock = asyncio.Lock()

    async def get_channel(self, session_id: str) -> SessionChannel:
        async with self._lock:
            channel = self._channels.get(session_id)
            if channel is None:
                channel = SessionChannel(buffer_size=self._buffer_size)
                self._channels[session_id] = channel
            return channel
