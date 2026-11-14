"""The synchronous publisher: the writer runs in a daemon thread."""

from __future__ import annotations

import atexit
import contextlib
import threading
import time
from collections.abc import Iterator
from time import monotonic
from types import TracebackType

from ._answer import NO_ANSWER, Answer, PublishResult, refusal
from ._core import PUB_TOKEN_TTL, Core
from ._errors import IQError
from ._frame import frame_size
from ._protocol import Outcome, check_outcome, check_seconds
from ._transport import Watch
from ._writer import OUT_OF_BAND, Command, Idle, Pause, Stream, Writer, drive

#: How often a held request that has nothing to write looks for an answer the server gave meanwhile.
ANSWER_POLL = 1.0

_open: set[Publisher] = set()
_open_lock = threading.Lock()
_hooked = False


def close_open_publishers() -> None:
    """The ``atexit`` hook: a daemon thread dies with the interpreter, so every publisher still open is closed
    with ``fail`` first."""
    with _open_lock:
        publishers = list(_open)
    for publisher in publishers:
        publisher._begin_close("fail")  # pyright: ignore[reportPrivateUsage]
    for publisher in publishers:
        with contextlib.suppress(Exception):
            publisher.close("fail")


class Publisher:
    """One stream's publisher, for synchronous code and thread pools.

    Created by ``IQClient.publisher()``; its request to the server opens at once, in a daemon thread that is the
    stream's only writer. ``send()`` may be called from any thread, and from inside a running event loop: it only
    puts a line on a queue. ``close()`` waits for the server's answer; async code calls it through
    ``asyncio.to_thread``.
    """

    def __init__(self, core: Core, stream_id: str) -> None:
        global _hooked
        self._stream_id = stream_id
        self._settings = core.settings
        self._transport = core.sync_transport()
        self._mint = core.mint
        self._wakeup = threading.Event()
        self._stopped = threading.Event()
        self._done = threading.Event()
        self._watch_lock = threading.Lock()
        self._watch: Watch | None = None
        self._close_lock = threading.Lock()
        self._close_deadline: float | None = None
        self._writer = Writer(
            core.settings, stream_id, wake=self._wakeup.set, stopping=self._stopping, settled=self._settled
        )
        with _open_lock:
            _open.add(self)
            if not _hooked:
                atexit.register(close_open_publishers)
                _hooked = True
        # Opened now, not on the first frame: a subscriber waits only PENDING_TIMEOUT_MS for its publisher.
        self._thread = threading.Thread(target=self._drive, name=f"iq-stream {stream_id}", daemon=True)
        self._thread.start()

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def max_frame_bytes(self) -> int:
        """The limit one line must fit, envelope and escapes included."""
        return self._settings.max_frame_bytes

    @property
    def frames_sent(self) -> int:
        """Data frames queued so far."""
        return self._writer.frames_sent

    @property
    def bytes_sent(self) -> int:
        """Bytes of the lines queued so far, newlines included."""
        return self._writer.bytes_sent

    @property
    def requests(self) -> int:
        """Requests opened for this stream: 1 unless one was lost and the stream continued in another."""
        return self._writer.requests

    @property
    def alive(self) -> bool:
        """False once the publisher was closed, aborted, or gave up."""
        return self._writer.alive

    @property
    def error(self) -> IQError | None:
        """Why the publisher gave up, when it did."""
        return self._writer.error

    def frame_size(self, data: str, event: str | None = None) -> int:
        """The size a frame would have on the wire, envelope and escapes included, without sending it."""
        return frame_size(data, event)

    def send(self, data: str, event: str | None = None) -> bool:
        """Queues one frame and returns whether the publisher is alive. Never waits for the network.

        Raises ``FrameTooLarge`` for a line over ``max_frame_bytes`` and ``ValueError`` for data that is not a
        string or an invalid event name. With ``on_error="raise"``, raises the error the publisher gave up on.
        """
        return self._writer.send(data, event)

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        """Sends the close line and returns the server's answer. The close line waits until the publisher is
        ``min_open`` old; ``timeout`` counts from then. The first outcome wins, and every call returns the same
        result."""
        deadline = self._begin_close(check_outcome(outcome), check_seconds("timeout", timeout))
        if not self._done.wait(max(0.0, deadline - monotonic())):
            self._writer.give_up(
                refusal(self._stream_id, "the close was not answered in time", NO_ANSWER, "close_timeout"),
                None,
                close_it=True,
            )
            self._done.wait(OUT_OF_BAND + 1.0)
        result = self._writer.settled_result()
        error = self._writer.error
        if error is not None and self._settings.raise_errors:
            raise error.with_traceback(None)
        return result

    def abort(self) -> None:
        """Drops the connection without closing the stream, for a process that hands the stream to another."""
        self._writer.abort()

    def __enter__(self) -> Publisher:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_type is None:
            self.close("end")
            return
        # The block's own exception is the one to see.
        with contextlib.suppress(IQError):
            self.close("fail")

    def _begin_close(self, outcome: Outcome, timeout: float = 10.0) -> float:
        """Queues the close line, once, and returns the deadline of its answer."""
        with self._close_lock:
            if self._close_deadline is None:
                self._close_deadline = self._writer.request_close(outcome) + timeout
            return self._close_deadline

    def __repr__(self) -> str:
        return f"<Publisher {self._stream_id} alive={self.alive} frames_sent={self.frames_sent}>"

    def _settled(self) -> None:
        self._done.set()
        with _open_lock:
            _open.discard(self)

    def _stopping(self) -> None:
        self._stopped.set()
        with self._watch_lock:
            watch = self._watch
        if watch is not None:
            watch.destroy()
        self._wakeup.set()

    def _drive(self) -> None:
        try:
            drive(self._writer.run(), self._carry_out)
        except Exception as cause:
            error = IQError(
                f"IQ stream {self._stream_id}: the publisher stopped on an internal error", stream_id=self._stream_id
            )
            error.__cause__ = cause
            self._writer.give_up(error, None, close_it=True)
        try:
            drive(self._writer.finish(), self._carry_out)
        except Exception:
            pass
        finally:
            self._writer.settled_result()

    def _carry_out(self, command: Command) -> Answer:
        if isinstance(command, Pause):
            if command.interruptible:
                self._stopped.wait(command.seconds)
            else:
                time.sleep(command.seconds)
            return NO_ANSWER
        watch = Watch(self._wakeup.set)
        with self._watch_lock:
            # A stop drops the connection in use. The requests that close a stream after a stop are not dropped.
            if not self._writer.stopped:
                self._watch = watch
            elif isinstance(command, Stream):
                return NO_ANSWER
        try:
            token = self._mint("pub", self._stream_id, PUB_TOKEN_TTL)
            if isinstance(command, Stream):
                return self._transport.stream(self._stream_id, token, self._body(watch), watch)
            return self._transport.post(self._stream_id, token, command.body, command.timeout, watch)
        finally:
            with self._watch_lock:
                if self._watch is watch:
                    self._watch = None

    def _body(self, watch: Watch) -> Iterator[bytes]:
        program = self._writer.body()
        try:
            for command in program:
                if isinstance(command, Idle):
                    self._wakeup.wait(min(command.seconds, ANSWER_POLL))
                    self._wakeup.clear()
                    if watch.answered():
                        return
                elif watch.answered():
                    # Once the server has answered it reads nothing more: what was taken is resent from the ring.
                    return
                else:
                    yield command.data
        finally:
            program.close()
