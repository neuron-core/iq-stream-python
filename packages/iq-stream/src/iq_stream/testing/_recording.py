from __future__ import annotations

import contextlib
import threading
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


class _Recorder:
    """What the two recording publishers share: everything but ``close()`` and the context manager."""

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
        #: How many times ``close()`` was called (awaited, for the asynchronous one), whatever it did.
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
    def on_error(self) -> Literal["report", "raise"]:
        return "raise" if self._raise else "report"

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

    def _close(self, outcome: Outcome) -> PublishResult:
        outcome = check_outcome(outcome)
        with self._lock:
            self.close_calls += 1
            if self.alive:
                self.outcome = outcome
            error = self.error
        if error is not None and self._raise:
            raise error.with_traceback(None)
        if self.outcome is None:
            return PublishResult(status=0, error=None if error is None else error.code)
        return PublishResult(
            status=200,
            last_event_id=len(self.frames),
            closed=True,
            outcome=self.outcome,
            reason="publisher" if self.outcome == "fail" else None,
        )


class RecordingPublisher(_Recorder):
    """``Publisher``'s surface in memory, for an adapter's unit tests.

    It validates frames as the real publisher does and records them, from any number of threads, in the order the
    ``send()`` calls returned. ``close()`` records the outcome; the first one wins, as on the wire. ``fail()`` makes
    it give up as the real one does after a fatal answer: ``send()`` then returns ``False``, or raises the error with
    ``on_error="raise"``. Like ``Publisher``, it is a synchronous context manager; asyncio code takes
    ``AsyncRecordingPublisher``.
    """

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        return self._close(outcome)

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


class AsyncRecordingPublisher(_Recorder):
    """``AsyncPublisher``'s surface in memory: the same as ``RecordingPublisher``, except that ``close()`` is a
    coroutine, as the real one is, and records the outcome only once it is awaited, so that a test shows a close that
    was never awaited. It is an asynchronous context manager."""

    async def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        return self._close(outcome)

    async def __aenter__(self) -> AsyncRecordingPublisher:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_type is None:
            await self.close("end")
            return
        with contextlib.suppress(IQError):
            await self.close("fail")
