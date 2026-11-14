from __future__ import annotations

import contextlib
import threading
from collections.abc import Generator
from dataclasses import dataclass
from types import TracebackType
from typing import Literal

from .._answer import PublishResult
from .._errors import IQError
from .._frame import checked_frame, frame_size
from .._protocol import DEFAULT_MAX_FRAME_BYTES, Outcome, check_outcome, check_stream_id
from .._token import new_stream_id


@dataclass(frozen=True)
class RecordedFrame:
    data: str
    event: str | None = None


@dataclass(frozen=True, eq=False)
class RecordedResult(PublishResult):
    """A ``PublishResult`` that can also be awaited, so that one recording publisher stands in for ``Publisher``
    (``pub.close()``) and for ``AsyncPublisher`` (``await pub.close()``)."""

    def __await__(self) -> Generator[None, None, PublishResult]:
        return self._itself().__await__()

    async def _itself(self) -> PublishResult:
        return self


class RecordingPublisher:
    """The publisher's surface in memory, for an adapter's unit tests.

    It validates frames as the real publishers do and records them, from any number of threads, in the order the
    ``send()`` calls returned. ``close()`` records the outcome; the first one wins, as on the wire. It works as a
    synchronous and as an asynchronous context manager, and ``close()`` can be called or awaited. ``fail()`` makes
    it give up as the real one does after a fatal answer: ``send()`` then returns ``False``, or raises the error
    with ``on_error="raise"``.
    """

    def __init__(
        self,
        stream_id: str | None = None,
        *,
        max_frame_bytes: int = DEFAULT_MAX_FRAME_BYTES,
        on_error: Literal["report", "raise"] = "report",
    ) -> None:
        self._stream_id = check_stream_id(new_stream_id() if stream_id is None else stream_id)
        self._max_frame_bytes = max_frame_bytes
        self._raise = on_error == "raise"
        self._lock = threading.Lock()
        #: The frames sent, in the order the calls returned.
        self.frames: list[RecordedFrame] = []
        #: The close line, once ``close()`` was called.
        self.outcome: Outcome | None = None
        self.aborted = False
        #: The error given to ``fail()``.
        self.error: IQError | None = None
        self.bytes_sent = 0
        #: How many times ``close()`` was called, whatever it did.
        self.close_calls = 0

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def max_frame_bytes(self) -> int:
        return self._max_frame_bytes

    @property
    def frames_sent(self) -> int:
        return len(self.frames)

    @property
    def requests(self) -> int:
        return 1

    @property
    def alive(self) -> bool:
        return self.error is None and self.outcome is None and not self.aborted

    @property
    def data(self) -> list[str]:
        """The data of every frame, in order."""
        return [frame.data for frame in self.frames]

    def frame_size(self, data: str, event: str | None = None) -> int:
        return frame_size(data, event)

    def send(self, data: str, event: str | None = None) -> bool:
        line = checked_frame(data, event, self._max_frame_bytes)
        with self._lock:
            if self.error is not None and self._raise:
                raise self.error.with_traceback(None)
            if not self.alive:
                return False
            self.frames.append(RecordedFrame(data, event))
            self.bytes_sent += len(line)
        return True

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> RecordedResult:
        outcome = check_outcome(outcome)
        with self._lock:
            self.close_calls += 1
            if self.alive:
                self.outcome = outcome
            error = self.error
        if error is not None and self._raise:
            raise error.with_traceback(None)
        if self.outcome is None:
            return RecordedResult(status=0, error=None if error is None else error.code)
        return RecordedResult(
            status=200,
            last_event_id=len(self.frames),
            closed=True,
            outcome=self.outcome,
            reason="publisher" if self.outcome == "fail" else None,
        )

    def abort(self) -> None:
        with self._lock:
            if self.alive:
                self.aborted = True

    def fail(self, error: IQError | None = None) -> None:
        """Makes the publisher give up, as the real one does after a fatal answer."""
        with self._lock:
            if self.alive:
                self.error = error or IQError(
                    f"IQ stream {self._stream_id}: the test made the publisher give up (test)",
                    stream_id=self._stream_id,
                    code="test",
                )

    def __enter__(self) -> RecordingPublisher:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close("end")
            return
        with contextlib.suppress(IQError):
            self.close("fail")

    async def __aenter__(self) -> RecordingPublisher:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        self.__exit__(exc_type, exc, traceback)
