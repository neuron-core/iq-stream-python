"""The synchronous publisher: the writer runs in a daemon thread."""

from __future__ import annotations

import atexit
import contextlib
import os
import threading
import time
from collections.abc import Iterator
from time import monotonic
from types import TracebackType
from typing import Literal

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


def _forget_the_parents_publishers() -> None:
    # A forked child has none of the parent's writer threads, and the connections are the parent's: the child's exit
    # leaves the parent's publishers alone. The lock may have been held by one of the parent's threads at the fork.
    global _open_lock
    _open_lock = threading.Lock()
    _open.clear()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_forget_the_parents_publishers)


def close_open_publishers(timeout: float = 10.0) -> None:
    """The ``atexit`` hook: a daemon thread dies with the interpreter, so every publisher still open is closed
    with ``fail`` first. They are closed side by side: exit takes one ``timeout`` and one out-of-band close, however
    many there are."""
    with _open_lock:
        publishers = list(_open)
    deadlines = [publisher._begin_close("fail", timeout) for publisher in publishers]  # pyright: ignore[reportPrivateUsage]
    for publisher, deadline in zip(publishers, deadlines, strict=True):
        publisher._done.wait(max(0.0, deadline - monotonic()))  # pyright: ignore[reportPrivateUsage]
    late = [publisher for publisher in publishers if not publisher._done.is_set()]  # pyright: ignore[reportPrivateUsage]
    for publisher in late:
        publisher._time_out()  # pyright: ignore[reportPrivateUsage]
    deadline = monotonic() + OUT_OF_BAND + 1.0
    for publisher in late:
        publisher._done.wait(max(0.0, deadline - monotonic()))  # pyright: ignore[reportPrivateUsage]


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
    def on_error(self) -> Literal["report", "raise"]:
        """The client's ``on_error``: whether ``send()`` and ``close()`` raise the error the publisher gave up on."""
        return "raise" if self._settings.raise_errors else "report"

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
        # On the writer's thread, from the error handler, the result is known already, and only this thread could
        # settle one: waiting would wait for itself.
        if threading.current_thread() is not self._thread and not self._done.wait(max(0.0, deadline - monotonic())):
            self._time_out()
            self._done.wait(OUT_OF_BAND + 1.0)
        result = self._writer.settled_result()
        error = self._writer.error
        if error is not None and self._settings.raise_errors:
            raise error.with_traceback(None)
        return result

    def abort(self) -> None:
        """Drops the connection and leaves the stream open, without a close line: it ends with the server's idle
        timeout, or with ``IQClient.close_stream()`` from another process."""
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

    def _time_out(self) -> None:
        self._writer.give_up(
            refusal(self._stream_id, "the close was not answered in time", NO_ANSWER, "close_timeout"),
            None,
            close_it=True,
        )

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
        program = self._writer.body(watch.answered)
        try:
            for command in program:
                if isinstance(command, Idle):
                    self._wakeup.wait(min(command.seconds, ANSWER_POLL))
                    self._wakeup.clear()
                else:
                    yield command.data
        finally:
            program.close()
