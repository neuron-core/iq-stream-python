"""The publisher's surface as protocols, for code that takes a publisher: the real ones and the testing kit's
recorders all fit, so an adapter types its parameters with these and its tests pass a recorder."""

from __future__ import annotations

from types import TracebackType
from typing import Literal, Protocol

from ._answer import PublishResult
from ._errors import IQError
from ._protocol import Outcome


class _Surface(Protocol):
    @property
    def stream_id(self) -> str: ...

    @property
    def max_frame_bytes(self) -> int: ...

    @property
    def frames_sent(self) -> int: ...

    @property
    def bytes_sent(self) -> int: ...

    @property
    def requests(self) -> int: ...

    @property
    def on_error(self) -> Literal["report", "raise"]: ...

    @property
    def alive(self) -> bool: ...

    @property
    def error(self) -> IQError | None: ...

    def frame_size(self, data: str, event: str | None = None) -> int: ...

    def send(self, data: str, event: str | None = None) -> bool: ...

    def abort(self) -> None: ...


class PublisherLike(_Surface, Protocol):
    """A publisher for synchronous code: ``Publisher``, or ``iq_stream.testing.RecordingPublisher``."""

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult: ...

    def __enter__(self) -> PublisherLike: ...

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None: ...


class AsyncPublisherLike(_Surface, Protocol):
    """A publisher for asyncio: ``AsyncPublisher``, or ``iq_stream.testing.AsyncRecordingPublisher``."""

    async def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult: ...

    async def __aenter__(self) -> AsyncPublisherLike: ...

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None: ...
