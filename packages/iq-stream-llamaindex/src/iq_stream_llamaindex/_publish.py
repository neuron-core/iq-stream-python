"""The one consumer of ``handler.stream_events()``: each event becomes frames on the publisher, the run's result is
awaited, the application saves the answer, and only then does the close line go out."""

from __future__ import annotations

import asyncio
import contextlib
import functools
import inspect
import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from dataclasses import replace
from types import TracebackType
from typing import TypeVar

from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, Outcome, PublishResult
from workflows.errors import WorkflowCancelledByUser, WorkflowRuntimeError
from workflows.events import Event, StopEvent
from workflows.handler import WorkflowHandler

from ._outcome import RunOutcome, classify, close_line
from .encoders.base import Encoder, Frame
from .encoders.native import NativeEncoder

logger = logging.getLogger("iq_stream_llamaindex")

#: Seconds the stream may take to deliver its terminal event once the run is done. A run killed with the
#: deprecated ``handler.cancel()`` never publishes one, and its stream would wait for ever.
GRACE = 1.0
#: Seconds given to a run to settle after its terminal event, or after ``cancel_run()``.
SETTLE = 10.0

CONSUMED = (
    "The handler's events were already read: handler.stream_events() has a single consumer. Pass the handler to "
    "publish() before anything reads its events, and read them from published() if the application needs them too."
)

EventHook = Callable[[Event], "Awaitable[None] | None"]
CloseHook = Callable[[RunOutcome], "Awaitable[None] | None"]

# The loop keeps tasks weakly: a cleanup that outlives a second cancellation must not be collected.
_cleanups: set[asyncio.Task[None]] = set()


class PublishedStream:
    """A run's events, each published before it is handed on: ``published()`` returns one.

    ``async with`` it and iterate it. The start frames go out on entering; every event of the stream is encoded and
    sent, then yielded, the terminal one last. Leaving the block normally reads and publishes the rest of the run,
    waits for its result, calls ``before_close``, sends the terminal frames and closes the stream. Leaving it with an
    exception, a cancellation included, stops the run (``cancel_run()``), sends the encoder's abort frames and closes
    the stream with ``fail``. ``outcome`` is set once the block has exited.
    """

    def __init__(
        self,
        handler: WorkflowHandler,
        publisher: AsyncPublisherLike,
        *,
        encoder: Encoder | None = None,
        before_close: CloseHook | None = None,
        cancelled_close: Outcome = "fail",
    ) -> None:
        if cancelled_close not in ("end", "fail"):
            raise ValueError('cancelled_close is "end" or "fail"')
        self._handler = handler
        self._publisher = publisher
        self._encoder: Encoder = encoder if encoder is not None else NativeEncoder()
        self._before_close = before_close
        self._cancelled_close: Outcome = cancelled_close
        self._entered = False
        self._events: AsyncGenerator[Event, None] | None = None
        self._result: asyncio.Task[StopEvent] | None = None
        self._reading: asyncio.Task[Event | None] | None = None
        self._read_any = False
        self._ended = False
        self._terminal: StopEvent | None = None
        self._live = True
        self._frames = 0
        self._outcome: RunOutcome | None = None

    @property
    def outcome(self) -> RunOutcome:
        """How the run ended; available once the ``async with`` block has exited."""
        if self._outcome is None:
            raise RuntimeError("The run has not ended: read the outcome after the async with block.")
        return self._outcome

    async def __aenter__(self) -> PublishedStream:
        if self._entered:
            raise RuntimeError("A PublishedStream is entered once.")
        self._entered = True
        self._result = asyncio.ensure_future(self._handler.stop_event_result())
        self._events = self._handler.stream_events()
        try:
            self._send(self._encoded(lambda: self._encoder.start(self._handler.run_id), "the start"), "the start")
        except BaseException as error:
            await self._abandon(error)
            raise
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, traceback: TracebackType | None
    ) -> None:
        if exc is not None:
            await self._abandon(exc)
            return
        try:
            while not self._ended:
                await self._next()
            await self._finish()
        except BaseException as error:
            if self._outcome is None:
                await self._abandon(error)
            raise

    def __aiter__(self) -> PublishedStream:
        return self

    async def __anext__(self) -> Event:
        if not self._entered:
            raise RuntimeError("Iterate the stream inside its block: async with published(...) as events.")
        if self._ended:
            raise StopAsyncIteration
        event = await self._next()
        if event is None:
            raise StopAsyncIteration
        return event

    async def _next(self) -> Event | None:
        """Reads one event and publishes it; the terminal event is held back until the answer is saved."""
        event = await self._read()
        if event is None:
            self._ended = True
            return None
        self._read_any = True
        if isinstance(event, StopEvent):
            self._terminal = event
            self._ended = True
        else:
            what = type(event).__name__
            self._send(self._encoded(lambda: self._encoder.encode(event), what), what)
        return event

    async def _read(self) -> Event | None:
        """The next event, or ``None`` when the stream ended, or when the run is done and the stream stays silent
        past the grace period."""
        assert self._events is not None and self._result is not None
        reading = self._reading = asyncio.ensure_future(_next_event(self._events))
        try:
            done, _ = await asyncio.wait((reading, self._result), return_when=asyncio.FIRST_COMPLETED)
            if reading not in done:
                await asyncio.wait((reading,), timeout=GRACE)
            if not reading.done():
                reading.cancel()
                await asyncio.wait((reading,))
                return None
            return reading.result()
        except WorkflowRuntimeError as error:
            if self._read_any:
                raise
            raise RuntimeError(CONSUMED) from error

    async def _finish(self) -> None:
        """The run's stream has ended: settle the result, persist, send the terminal frames, close."""
        handler = self._handler
        if self._terminal is None and not handler.is_done():
            await handler.cancel_run()
        stop_event, error = await self._settle()
        terminal = self._terminal
        status = classify(terminal, error)
        result = None
        if error is None and stop_event is not None:
            result = stop_event.result if type(stop_event) is StopEvent else stop_event
        outcome = RunOutcome(
            status=status,
            run_id=handler.run_id,
            result=result,
            stop_event=terminal or stop_event or handler.get_stop_event(),
            error=error,
            frames=self._frames,
            published=self._live,
        )
        if self._before_close is not None:
            try:
                await _call(self._before_close, outcome)
            except BaseException as failure:
                # The answer is not in the application's store: the stream must not say it finished.
                self._send(
                    self._encoded(functools.partial(self._encoder.abort, failure), "the abort"), "the abort", quiet=True
                )
                answer = await self._close("fail", quiet=True)
                self._outcome = replace(outcome, frames=self._frames, published=self._live, publish_result=answer)
                raise
        if terminal is not None:
            frames = self._encoded(lambda: self._encoder.encode(terminal), type(terminal).__name__)
        else:
            reason = asyncio.CancelledError() if status == "cancelled" else error or RuntimeError("no terminal event")
            frames = self._encoded(lambda: self._encoder.abort(reason), "the abort")
        self._send(frames, "the end")
        answer = await self._close(close_line(status, self._cancelled_close))
        published = self._live and (answer is None or answer.error is None)
        self._outcome = replace(outcome, frames=self._frames, published=published, publish_result=answer)

    async def _settle(self) -> tuple[StopEvent | None, BaseException | None]:
        """The handler's stop event, or what awaiting it raises. A result task that was cancelled outright (the
        deprecated hard cancel) counts as a cancelled run."""
        assert self._result is not None
        task = self._result
        await asyncio.wait((task,), timeout=SETTLE)
        if not task.done():
            task.cancel()
            await asyncio.wait((task,))
        if task.cancelled():
            return None, WorkflowCancelledByUser("The run was stopped without a terminal event.")
        error = task.exception()
        return (None, error) if error is not None else (task.result(), None)

    async def _abandon(self, error: BaseException) -> None:
        """The block raised, or the task was cancelled: stop the run and close with ``fail``. The cleanup is shielded,
        so that a second cancellation does not leave the stream open."""
        if self._outcome is not None:
            return
        cleanup = asyncio.ensure_future(self._cleanup(error))
        _cleanups.add(cleanup)
        cleanup.add_done_callback(_cleanups.discard)
        await asyncio.shield(cleanup)

    async def _cleanup(self, error: BaseException) -> None:
        reading = self._reading
        if reading is not None and not reading.done():
            reading.cancel()
            await asyncio.wait((reading,))
        if self._events is not None:
            with contextlib.suppress(Exception):
                await self._events.aclose()
        handler = self._handler
        if not handler.is_done():
            with contextlib.suppress(Exception):
                await handler.cancel_run()
        stop_event, run_error = (None, None) if self._result is None else await self._settle()
        self._send(self._encoded(lambda: self._encoder.abort(error), "the abort"), "the abort", quiet=True)
        published = self._live
        answer = await self._close("fail", quiet=True)
        self._outcome = RunOutcome(
            status="cancelled" if isinstance(error, asyncio.CancelledError) else classify(self._terminal, run_error),
            run_id=handler.run_id,
            result=None,
            stop_event=self._terminal or stop_event,
            error=run_error or error,
            frames=self._frames,
            published=published and (answer is None or answer.error is None),
            publish_result=answer,
        )

    async def _close(self, outcome: Outcome, *, quiet: bool = False) -> PublishResult | None:
        """Closes the stream; nothing when the publisher gave up, since it closed the stream itself."""
        if not (self._live and self._publisher.alive):
            self._live = False
            return None
        if not quiet:
            return await self._publisher.close(outcome)
        try:
            return await self._publisher.close(outcome)
        except IQError:
            self._live = False
            return None

    def _encoded(self, produce: Callable[[], Iterable[Frame]], what: str) -> list[Frame]:
        """What the encoder makes; nothing once the publisher gave up, and nothing (with a warning) when the encoder
        raises: one event the encoder cannot handle must not end the stream."""
        if not self._live:
            return []
        try:
            return list(produce())
        except Exception as error:
            logger.warning(
                "IQ stream %s: the encoder failed on %s (%s); its frames were skipped",
                self._publisher.stream_id,
                what,
                type(error).__name__,
            )
            return []

    def _fits(self, frame: Frame) -> bool:
        return self._publisher.frame_size(frame[1], frame[0]) <= self._publisher.max_frame_bytes

    def _send(self, frames: Iterable[Frame], what: str, *, quiet: bool = False) -> None:
        """Sends frames in order until the publisher gives up. ``quiet`` swallows the publisher's own error (with
        ``on_error="raise"``) on the way out of a run that already failed."""
        try:
            for frame in frames:
                if not self._live:
                    return
                self._send_frame(frame, what)
        except IQError:
            if not quiet:
                raise
            self._live = False

    def _send_frame(self, frame: Frame, what: str) -> None:
        try:
            self._count(self._publisher.send(frame[1], frame[0]))
        except FrameTooLarge as error:
            self._send_fitted(frame, what, error)
        except ValueError as error:
            logger.warning(
                "IQ stream %s: a frame of %s was refused (%s) and dropped",
                self._publisher.stream_id,
                what,
                type(error).__name__,
            )

    def _send_fitted(self, frame: Frame, what: str, error: FrameTooLarge) -> None:
        """A frame over the line limit: what the encoder makes of it instead, or nothing."""
        try:
            fitted = list(self._encoder.fit(frame, self._fits))
        except Exception:
            fitted = []
        if not fitted or not all(self._fits(piece) for piece in fitted):
            logger.warning(
                "IQ stream %s: dropped a frame of %s: %d bytes, over the limit of %d",
                self._publisher.stream_id,
                what,
                error.size,
                error.limit,
            )
            return
        for piece in fitted:
            if not self._live:
                return
            self._count(self._publisher.send(piece[1], piece[0]))

    def _count(self, alive: bool) -> None:
        if alive:
            self._frames += 1
        else:
            self._live = False


def published(
    handler: WorkflowHandler,
    publisher: AsyncPublisherLike,
    *,
    encoder: Encoder | None = None,
    before_close: CloseHook | None = None,
    cancelled_close: Outcome = "fail",
) -> PublishedStream:
    """Publishes a run and hands each event on to the application, for code that still reads the events (to answer a
    human-input request, to log):

    .. code-block:: python

        async with published(handler, publisher, encoder=AGUIEncoder(thread_id=thread_id)) as events:
            async for event in events:
                ...
        outcome = events.outcome

    Breaking out of the loop early does not cut the browser's stream short: leaving the block reads and publishes
    the rest of the run. Leaving it with an exception stops the run and closes the stream with ``fail``.
    """
    return PublishedStream(
        handler, publisher, encoder=encoder, before_close=before_close, cancelled_close=cancelled_close
    )


async def publish(
    handler: WorkflowHandler,
    publisher: AsyncPublisherLike,
    *,
    encoder: Encoder | None = None,
    on_event: EventHook | None = None,
    before_close: CloseHook | None = None,
    cancelled_close: Outcome = "fail",
    raise_run_errors: bool = False,
) -> RunOutcome:
    """Publishes a run to the browser and returns how it ended.

    It is the one reader of ``handler.stream_events()``: each event is encoded (``NativeEncoder`` by default) and
    sent, then given to ``on_event``. When the stream ends it awaits the run's result, calls ``before_close`` with the
    outcome (where the application saves the answer), sends the terminal frames and closes the stream: ``end`` for a
    completed run, ``fail`` for a failed or timed-out one, ``cancelled_close`` for a cancelled one.

    A failed run is returned, not raised, unless ``raise_run_errors`` is true: ``publish()`` usually runs in a task
    of its own. Exceptions from ``on_event`` and ``before_close`` propagate, the stream then closed with ``fail``. A
    cancellation of the task stops the run, closes the stream with ``fail`` and propagates. Publishing fails open: a
    publisher that gives up stops the encoding, and the run is still read to its end and ``before_close`` called.
    """
    stream = published(handler, publisher, encoder=encoder, before_close=before_close, cancelled_close=cancelled_close)
    async with stream:
        async for event in stream:
            if on_event is not None:
                await _call(on_event, event)
    outcome = stream.outcome
    if raise_run_errors and outcome.error is not None and outcome.status != "completed":
        raise outcome.error
    return outcome


async def _next_event(events: AsyncGenerator[Event, None]) -> Event | None:
    try:
        return await anext(events)
    except StopAsyncIteration:
        return None


_T = TypeVar("_T")


async def _call(hook: Callable[[_T], Awaitable[None] | None], value: _T) -> None:
    returned = hook(value)
    if inspect.isawaitable(returned):
        await returned
