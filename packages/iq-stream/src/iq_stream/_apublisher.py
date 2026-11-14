"""The asyncio publisher: the writer is a task on the running loop."""

from __future__ import annotations

import asyncio
import contextlib
import threading
import time
from collections.abc import AsyncIterator, Callable
from time import monotonic
from types import TracebackType

from ._answer import NO_ANSWER, Answer, PublishResult, refusal
from ._core import PUB_TOKEN_TTL, Core
from ._errors import IQError
from ._frame import frame_size
from ._protocol import Outcome, check_outcome, check_seconds
from ._publisher import ANSWER_POLL
from ._transport import Watch
from ._writer import OUT_OF_BAND, Command, Idle, Pause, Stream, Writer, adrive, drive

# The loop keeps only weak references to tasks: a writer must not be collected while it holds a stream open.
_tasks: set[asyncio.Task[object]] = set()


def _keep(task: asyncio.Task[object]) -> None:
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


class AsyncPublisher:
    """One stream's publisher, for asyncio.

    Created by ``IQClient.apublisher()`` on a running loop; its request opens when the loop next runs. ``send()``
    is a plain method, never awaited, and belongs to the loop's thread; only ``close()`` is awaited, and it is
    shielded from cancellation, since a cancelled run is exactly when the close line must still go out.
    """

    def __init__(self, core: Core, stream_id: str) -> None:
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            raise RuntimeError(
                "apublisher() is called from a coroutine on a running event loop; synchronous code uses publisher()."
            ) from None
        self._loop_thread = threading.get_ident()
        self._stream_id = stream_id
        self._settings = core.settings
        self._transport = core.async_transport()
        self._sync_transport = core.sync_transport
        self._mint = core.mint
        self._orphan_lock = threading.Lock()
        self._orphaned_at_all = False
        self._orphan_done = threading.Event()
        self._wakeup = asyncio.Event()
        self._stopped = asyncio.Event()
        self._done = asyncio.Event()
        self._watch_lock = threading.Lock()
        self._watch: Watch | None = None
        self._closer: asyncio.Task[PublishResult] | None = None
        self._writer = Writer(
            core.settings,
            stream_id,
            wake=self._wake,
            stopping=self._stopping,
            settled=lambda: self._on_loop(self._done.set),
        )
        self._task: asyncio.Task[None] = self._loop.create_task(self._drive(), name=f"iq-stream {stream_id}")
        _keep(self._task)  # pyright: ignore[reportArgumentType]

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
        """Queues one frame and returns whether the publisher is alive. Not awaited: nothing can come between
        deciding to send and the frame being in the queue.

        Raises ``FrameTooLarge`` for a line over ``max_frame_bytes`` and ``ValueError`` for data that is not a
        string or an invalid event name. With ``on_error="raise"``, raises the error the publisher gave up on.
        """
        if self._loop.is_closed():
            self._orphaned()
        return self._writer.send(data, event)

    async def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        """Sends the close line and returns the server's answer. The close line waits until the publisher is
        ``min_open`` old; ``timeout`` counts from then. The first outcome wins, and every call returns the same
        result. Cancelling the caller does not stop the close."""
        outcome = check_outcome(outcome)
        timeout = check_seconds("timeout", timeout)
        if self._loop.is_closed():
            # Closed by its loop's owner without asyncio.run()'s teardown: its writer will never run again.
            self._orphaned()
            await asyncio.to_thread(self._orphan_done.wait, OUT_OF_BAND + 1.0)
            result = self._writer.settled_result()
        else:
            if self._closer is None:
                deadline = self._writer.request_close(outcome) + timeout
                self._closer = self._loop.create_task(self._closing(deadline))
                _keep(self._closer)  # pyright: ignore[reportArgumentType]
            result = await asyncio.shield(self._closer)
        error = self._writer.error
        if error is not None and self._settings.raise_errors:
            raise error.with_traceback(None)
        return result

    def abort(self) -> None:
        """Drops the connection and leaves the stream open, without a close line: it ends with the server's idle
        timeout, or with ``IQClient.close_stream()`` from another process."""
        self._writer.abort()

    async def __aenter__(self) -> AsyncPublisher:
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc_type is None:
            await self.close("end")
            return
        # The block's own exception is the one to see.
        with contextlib.suppress(IQError):
            await self.close("fail")

    def __repr__(self) -> str:
        return f"<AsyncPublisher {self._stream_id} alive={self.alive} frames_sent={self.frames_sent}>"

    def _on_loop(self, callback: Callable[[], object]) -> None:
        if self._loop.is_closed():
            self._orphaned()
            return
        # A loop closed meanwhile raises RuntimeError: nothing waits on it any more.
        with contextlib.suppress(RuntimeError):
            if threading.get_ident() == self._loop_thread:
                callback()
            else:
                self._loop.call_soon_threadsafe(callback)

    def _orphaned(self) -> None:
        """The loop was closed under a writer that had not finished, so nothing will run it again: the publisher goes
        inert, and a thread of its own closes the stream out of band over the synchronous transport."""
        with self._orphan_lock:
            if self._orphaned_at_all:
                return
            self._orphaned_at_all = True
        self._writer.cancelled()
        thread = threading.Thread(target=self._finish_without_loop, name=f"iq-stream {self._stream_id}", daemon=True)
        try:
            thread.start()
        except RuntimeError:
            # At the interpreter's shutdown no thread starts, and the stream ends with the server's idle timeout.
            self._orphan_done.set()

    def _finish_without_loop(self) -> None:
        transport = self._sync_transport()

        def carry_out(command: Command) -> Answer:
            if isinstance(command, Pause):
                time.sleep(command.seconds)
                return NO_ANSWER
            if isinstance(command, Stream):
                return NO_ANSWER
            token = self._mint("pub", self._stream_id, PUB_TOKEN_TTL)
            return transport.post(self._stream_id, token, command.body, command.timeout, Watch())

        try:
            drive(self._writer.finish(), carry_out)
        except Exception:
            pass
        finally:
            self._writer.settled_result()
            self._orphan_done.set()

    def _wake(self) -> None:
        self._on_loop(self._wakeup.set)

    def _stopping(self) -> None:
        self._on_loop(self._stopped.set)
        with self._watch_lock:
            watch = self._watch
        if watch is not None:
            watch.destroy()
        self._wake()

    async def _wait(self, event: asyncio.Event, seconds: float) -> bool:
        try:
            await asyncio.wait_for(event.wait(), max(0.0, seconds))
        except asyncio.TimeoutError:
            return False
        return True

    async def _closing(self, deadline: float) -> PublishResult:
        if not await self._wait(self._done, deadline - monotonic()):
            self._writer.give_up(
                refusal(self._stream_id, "the close was not answered in time", NO_ANSWER, "close_timeout"),
                None,
                close_it=True,
            )
            await self._wait(self._done, OUT_OF_BAND + 1.0)
        return self._writer.settled_result()

    async def _drive(self) -> None:
        cancelled: asyncio.CancelledError | None = None
        try:
            try:
                await adrive(self._writer.run(), self._carry_out)
            except asyncio.CancelledError as cancel:
                # The loop is shutting down, or someone cancelled every task: close the stream before going.
                cancelled = cancel
                self._writer.cancelled()
            except Exception as cause:
                error = IQError(
                    f"IQ stream {self._stream_id}: the publisher stopped on an internal error",
                    stream_id=self._stream_id,
                )
                error.__cause__ = cause
                self._writer.give_up(error, None, close_it=True)
            await adrive(self._writer.finish(), self._carry_out)
        except Exception:
            pass
        finally:
            self._writer.settled_result()
            with contextlib.suppress(Exception):
                await self._transport.aclose()
        if cancelled is not None:
            raise cancelled

    async def _carry_out(self, command: Command) -> Answer:
        if isinstance(command, Pause):
            if command.interruptible:
                await self._wait(self._stopped, command.seconds)
            else:
                await asyncio.sleep(command.seconds)
            return NO_ANSWER
        watch = Watch(self._wake)
        with self._watch_lock:
            # A stop drops the connection in use. The requests that close a stream after a stop are not dropped.
            if not self._writer.stopped:
                self._watch = watch
            elif isinstance(command, Stream):
                return NO_ANSWER
        try:
            token = self._mint("pub", self._stream_id, PUB_TOKEN_TTL)
            if isinstance(command, Stream):
                return await self._transport.stream(self._stream_id, token, self._body(watch), watch)
            return await self._transport.post(self._stream_id, token, command.body, command.timeout, watch)
        finally:
            with self._watch_lock:
                if self._watch is watch:
                    self._watch = None

    async def _idle(self, seconds: float) -> None:
        # A timer that sets the wake-up event: cheaper than a task per wait, and a spurious wake-up is harmless.
        if not self._wakeup.is_set():
            timer = self._loop.call_later(max(0.0, seconds), self._wakeup.set)
            try:
                await self._wakeup.wait()
            finally:
                timer.cancel()
        self._wakeup.clear()

    async def _body(self, watch: Watch) -> AsyncIterator[bytes]:
        program = self._writer.body(watch.answered)
        try:
            for command in program:
                if isinstance(command, Idle):
                    await self._idle(min(command.seconds, ANSWER_POLL))
                else:
                    yield command.data
        finally:
            program.close()
