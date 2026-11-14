"""What the unit tests share: a client whose publishers record, and readers of what they recorded."""

from __future__ import annotations

import json
from collections.abc import AsyncIterable
from typing import Any, TypeVar

from iq_stream import IQClient, IQError
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher

T = TypeVar("T")

SECRET = "test-app-secret-0123456789abcdef0123"


class RecordingClient(IQClient):
    """A client whose publishers record in memory instead of opening a request."""

    def __init__(self, max_frame_bytes: int = 65_536) -> None:
        super().__init__("http://127.0.0.1:9", "demo-app", SECRET, public_url="https://stream.example.com")
        self.recorders: list[AsyncRecordingPublisher] = []
        self._limit = max_frame_bytes

    def apublisher(self, stream_id: str) -> Any:
        recorder = AsyncRecordingPublisher(stream_id, max_frame_bytes=self._limit)
        self.recorders.append(recorder)
        return recorder


def payloads(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[dict[str, Any]]:
    return [json.loads(data) for data in publisher.data]


def names(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[str]:
    """The Agno event names of a native stream."""
    return [payload["event"] for payload in payloads(publisher)]


def types(publisher: RecordingPublisher | AsyncRecordingPublisher, raw: bool = True) -> list[str]:
    """The AG-UI types of an AG-UI stream, ``RAW`` included or not."""
    return [payload["type"] for payload in payloads(publisher) if raw or payload["type"] != "RAW"]


async def collect(items: AsyncIterable[T]) -> list[T]:
    return [item async for item in items]


class Errors(list[IQError]):
    """An ``on_error`` that keeps what it was given."""

    def __call__(self, error: IQError) -> None:
        self.append(error)
