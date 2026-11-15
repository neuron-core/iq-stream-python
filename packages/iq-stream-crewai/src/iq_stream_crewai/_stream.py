from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import re
import threading
import uuid
from collections.abc import Callable, Collection, Iterable
from types import TracebackType
from typing import Any, Literal

from crewai.events.base_events import BaseEvent
from iq_stream import FrameTooLarge, IQClient, IQError, Outcome, PublisherLike, PublishResult

from ._agui import CANCELLATIONS, AgUiEncoder
from ._encoder import Encoder
from ._limits import Fitter, Frame
from ._native import NativeEncoder
from ._snapshot import Snapshotter, resolve_types
from ._transport import BusTransport, SinkTransport, Transport, choose

logger = logging.getLogger("iq_stream_crewai")

_STREAM_ID = re.compile(r"[A-Za-z0-9_-]{16,128}")

Protocol = Literal["ag-ui", "crewai"]


class CrewAIStream:
    """Everything CrewAI emits inside the block goes to one IQ stream. Made by ``publish()``.

    The block closes the stream with ``end`` when it exits normally and with ``fail`` when it exits with an
    exception, which then goes on; a cancellation closes it with ``close_on_cancel``. The publisher is created, and its
    request opened, when the block is entered.
    """

    def __init__(
        self,
        target: IQClient | PublisherLike,
        stream_id: str | None = None,
        *,
        protocol: Protocol | Encoder[Any] = "ag-ui",
        thread_id: str | None = None,
        run_id: str | None = None,
        include: Collection[str] | None = None,
        exclude: Collection[str] = (),
        max_field_bytes: int | None = 16_384,
        expose_errors: bool = False,
        close_on_cancel: Outcome = "fail",
        transport: Literal["auto", "sink", "bus"] = "auto",
    ) -> None:
        if isinstance(target, IQClient):
            self._client: IQClient | None = target
            self._given: PublisherLike | None = None
            resolved = target.new_stream_id() if stream_id is None else stream_id
        else:
            if inspect.iscoroutinefunction(getattr(target, "close", None)):
                raise TypeError("publish() takes a synchronous publisher: CrewAI emits from worker threads.")
            if stream_id is not None and stream_id != target.stream_id:
                raise ValueError("stream_id differs from the publisher's own.")
            self._client = None
            self._given = target
            resolved = target.stream_id
        if not isinstance(resolved, str) or not _STREAM_ID.fullmatch(resolved):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("Invalid stream id: 16 to 128 characters of A-Z a-z 0-9 _ -.")
        methods = ("snapshot", "start", "frames", "finish")
        if isinstance(protocol, str):
            if protocol not in ("ag-ui", "crewai"):
                raise ValueError("protocol is 'ag-ui', 'crewai' or an Encoder.")
        elif not all(callable(getattr(protocol, method, None)) for method in methods):
            raise ValueError("protocol is 'ag-ui', 'crewai' or an Encoder.")
        if close_on_cancel not in ("end", "fail"):
            raise ValueError("close_on_cancel is 'end' or 'fail'.")
        if transport not in ("auto", "sink", "bus"):
            raise ValueError("transport is 'auto', 'sink' or 'bus'.")
        if max_field_bytes is not None and (isinstance(max_field_bytes, bool) or max_field_bytes < 1):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise ValueError("max_field_bytes is a positive number of bytes, or None.")
        self._stream_id = resolved
        self._protocol = protocol
        self._thread_id = thread_id or resolved
        self._run_id = run_id or uuid.uuid4().hex
        self._types = resolve_types(include, exclude)
        self._max_field_bytes = max_field_bytes
        self._expose_errors = expose_errors
        self._close_on_cancel: Outcome = close_on_cancel
        self._transport_kind = choose(transport)

        self._lock = threading.Lock()
        self._errors_lock = threading.Lock()
        self._entered = False
        self._closed = False
        self._logged = False
        self._publisher: PublisherLike | None = None
        self._encoder: Encoder[Any] | None = None
        self._transport: Transport | None = None
        self._frames_sent = 0
        #: The adapter's own failures, which never break the run: a frame dropped for its size, an encoder error.
        #: The publisher's own are in ``publisher.error``.
        self.errors: list[BaseException] = []
        #: The server's answer to the close line, once the block has exited.
        self.result: PublishResult | None = None

    @property
    def stream_id(self) -> str:
        return self._stream_id

    @property
    def thread_id(self) -> str:
        """AG-UI's ``threadId``: the stream id unless one was given."""
        return self._thread_id

    @property
    def run_id(self) -> str:
        """AG-UI's ``runId``."""
        return self._run_id

    @property
    def publisher(self) -> PublisherLike | None:
        """The core's publisher, once the block has been entered."""
        return self._publisher

    @property
    def frames_sent(self) -> int:
        """Frames the publisher took, the application's own included."""
        return self._frames_sent

    def send(self, data: str, event: str | None = None) -> bool:
        """Sends a frame of the application's own, in order with the run's. Returns whether the publisher is alive;
        raises ``FrameTooLarge`` and ``ValueError`` as the publisher's ``send()`` does, and ``RuntimeError`` outside
        the block."""
        with self._lock:
            if self._publisher is None or self._closed:
                raise RuntimeError("send() is for inside the block.")
            alive = self._publisher.send(data, event)
            if alive:
                self._frames_sent += 1
            return alive

    # -- the events -------------------------------------------------------------------------------------------------

    def _snapshot(self, source: object, event: BaseEvent) -> object | None:
        """Runs in the thread that emitted the event. Never raises: an exception here would break the run."""
        try:
            publisher, encoder = self._publisher, self._encoder
            if self._closed or publisher is None or encoder is None or not publisher.alive:
                return None
            return encoder.snapshot(source, event)
        except Exception as error:
            self._fail(error)
            return None

    def _deliver(self, snapshot: object) -> None:
        try:
            with self._lock:
                # A thread started in the block keeps the sink in its copy of the context after the block exits.
                if self._closed or self._encoder is None:
                    return
                self._emit(self._encoder.frames(snapshot))
        except Exception as error:
            self._fail(error)

    def _emit(self, frames: Iterable[Frame]) -> None:
        """Hands frames to the publisher. Called with the lock held."""
        publisher = self._publisher
        assert publisher is not None
        for event, data in frames:
            try:
                if not publisher.send(data, event):
                    return  # it gave up: nobody will receive the rest
                self._frames_sent += 1
            except IQError as error:
                # FrameTooLarge from an encoder of the application's, or the publisher's own error with
                # on_error="raise": the run goes on, and close() raises that error at the end of the block.
                self._fail(error)
                if not isinstance(error, FrameTooLarge):
                    return

    def _fail(self, error: BaseException) -> None:
        with self._errors_lock:
            self.errors.append(error)
            if self._logged:
                return
            self._logged = True
        # The type only: a message may carry frame data.
        logger.warning(
            "iq-stream-crewai: stream %s: %s while publishing a CrewAI event; the run goes on and later failures "
            "are in CrewAIStream.errors",
            self._stream_id,
            type(error).__name__,
        )

    # -- the block --------------------------------------------------------------------------------------------------

    def _make_encoder(self, publisher: PublisherLike) -> Encoder[Any]:
        if not isinstance(self._protocol, str):
            return self._protocol
        snapshotter = Snapshotter(self._types, self._max_field_bytes, self._expose_errors)
        fitter = Fitter(publisher.max_frame_bytes, publisher.frame_size, self._fail)
        if self._protocol == "crewai":
            return NativeEncoder(snapshotter, fitter)
        return AgUiEncoder(snapshotter, fitter, self._thread_id, self._run_id, self._expose_errors)

    def _open(self) -> None:
        if self._entered:
            raise RuntimeError("A CrewAIStream is entered once; call publish() for each run.")
        self._entered = True
        if self._client is not None:
            publisher: PublisherLike = self._client.publisher(self._stream_id)
        else:
            assert self._given is not None
            publisher = self._given
        self._publisher = publisher
        self._encoder = self._make_encoder(publisher)
        with self._lock:
            try:
                self._emit(self._encoder.start())
            except Exception as error:
                self._fail(error)
        kind = SinkTransport if self._transport_kind == "sink" else BusTransport
        transport = kind(self._snapshot, self._deliver)
        try:
            transport.attach()
        except Exception as error:
            self._fail(error)  # the run goes on unpublished, and the stream is still closed
        else:
            self._transport = transport

    def _finish(self, error: BaseException | None) -> Outcome:
        """Sends the encoder's closing frames and marks the stream closed. The transport is detached already."""
        if self._transport is not None:
            try:
                self._transport.settle()
            except Exception as failure:
                self._fail(failure)
        with self._lock:
            self._closed = True
            if self._encoder is not None and self._publisher is not None and self._publisher.alive:
                try:
                    self._emit(self._encoder.finish(error))
                except Exception as failure:
                    self._fail(failure)
        if error is None or (isinstance(error, CANCELLATIONS) and self._close_on_cancel == "end"):
            return "end"
        return "fail"

    def _close(self, outcome: Outcome, raise_errors: bool) -> None:
        assert self._publisher is not None
        try:
            self.result = self._publisher.close(outcome)
        except IQError:
            # Only with on_error="raise", which asks for this: after a normal exit, the block raises it.
            if raise_errors:
                raise
        except Exception as error:
            self._fail(error)

    def __enter__(self) -> CrewAIStream:
        self._open()
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> Literal[False]:
        if self._transport is not None:
            self._transport.detach()
        self._close(self._finish(exc), raise_errors=exc is None)
        return False

    async def __aenter__(self) -> CrewAIStream:
        self._open()
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> Literal[False]:
        if self._transport is not None:
            self._transport.detach()  # in the task's own context, before anything is awaited

        raise_errors = exc is None
        if self._transport is not None and self._transport.blocking:
            # The bus is flushed before the closing frames, which blocks: all of it in a thread.
            await _in_thread(lambda: self._close(self._finish(exc), raise_errors))
        else:
            # The closing frames go out at once, even when the task is being cancelled; only the server's answer is
            # waited for, in a thread, and shielded so that a cancellation cannot drop the close line.
            outcome = self._finish(exc)
            await _in_thread(lambda: self._close(outcome, raise_errors))
        return False


async def _in_thread(work: Callable[[], None]) -> None:
    """Runs ``work`` in a thread of its own and waits for it, shielded from cancellation. Not the loop's default
    executor: ``kickoff_async`` holds one of its threads for a whole crew run, and a busy server would make the close
    line wait for other requests' crews."""
    loop = asyncio.get_running_loop()
    done: asyncio.Future[None] = loop.create_future()
    # Retrieved even when nobody waits any more (the task was cancelled meanwhile), so that asyncio logs nothing.
    done.add_done_callback(lambda future: future.cancelled() or future.exception())

    def settle(error: BaseException | None) -> None:
        if done.done():
            return
        if error is None:
            done.set_result(None)
        else:
            done.set_exception(error)

    def run() -> None:
        error: BaseException | None = None
        try:
            work()
        except BaseException as caught:
            error = caught
        with contextlib.suppress(RuntimeError):  # the loop is closed: nobody is waiting
            loop.call_soon_threadsafe(settle, error)

    threading.Thread(target=run, name="iq-stream-crewai close").start()
    await asyncio.shield(done)


def publish(
    target: IQClient | PublisherLike,
    stream_id: str | None = None,
    *,
    protocol: Protocol | Encoder[Any] = "ag-ui",
    thread_id: str | None = None,
    run_id: str | None = None,
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
    max_field_bytes: int | None = 16_384,
    expose_errors: bool = False,
    close_on_cancel: Outcome = "fail",
    transport: Literal["auto", "sink", "bus"] = "auto",
) -> CrewAIStream:
    """Publishes everything CrewAI emits inside a ``with`` (or ``async with``) block to one IQ stream.

    ``target`` is an ``IQClient``, whose publisher for ``stream_id`` (a new id when ``None``) is created when the block
    is entered, or a synchronous publisher the block takes over. Nothing is sent before the block is entered.

    - ``protocol``: ``"ag-ui"`` (AG-UI events, unnamed frames), ``"crewai"`` (CrewAI's own events, named after their
      type), or an ``Encoder``.
    - ``thread_id``, ``run_id``: AG-UI's ids of the run; by default the stream id and a new uuid.
    - ``include``, ``exclude``: the CrewAI event types the encoder sees (``EVENT_TYPES`` by default).
    - ``max_field_bytes``: tool results, task outputs and other text values are cut to this many bytes of UTF-8, with
      a marker; ``None`` cuts nothing. Frames are also cut, or split, to fit the publisher's ``max_frame_bytes``.
    - ``expose_errors``: publish error messages as they are; by default every one is "An error occurred.".
    - ``close_on_cancel``: the close line after a cancellation (``CancelledError``, ``KeyboardInterrupt``).
    - ``transport``: ``"sink"`` (a scoped stream sink, CrewAI 1.15.2 and later), ``"bus"`` (handlers on the global
      event bus: degraded, for CrewAI without stream sinks), or ``"auto"``.
    """
    return CrewAIStream(
        target,
        stream_id,
        protocol=protocol,
        thread_id=thread_id,
        run_id=run_id,
        include=include,
        exclude=exclude,
        max_field_bytes=max_field_bytes,
        expose_errors=expose_errors,
        close_on_cancel=close_on_cancel,
        transport=transport,
    )
