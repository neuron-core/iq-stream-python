"""The tee: every item of an Agno run's iterator goes on to the caller unchanged, and becomes frames for the publisher;
the stream is closed when the iterator ends, with the outcome the root run's events decide."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar, cast

from agno.run.base import BaseRunOutputEvent
from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, Outcome, PublisherLike, PublishResult

from ._common import (
    MASKED_ERROR,
    ROOT_CANCELS,
    ROOT_ENDINGS,
    ROOT_ERRORS,
    ROOT_PAUSES,
    AguiRequest,
    Frame,
    event_name,
    nested,
)
from ._common import Protocol as WireProtocol
from ._native import NativeEncoder, Transform, error_event

if TYPE_CHECKING:
    from ._agui import AguiEncoder

E = TypeVar("E")

RunStatus = Literal["completed", "paused", "error", "cancelled", "interrupted"]
"""How a run ended: Agno completed it, paused it for a human, failed it, cancelled it; or it was ``interrupted``,
because the consumer stopped iterating, its task was cancelled or the process was interrupted."""

logger = logging.getLogger("iq_stream_agno")

#: A run stopped from outside: its task cancelled, the process interrupted. Closed with ``close_on_cancel``.
_CANCELLATIONS = (asyncio.CancelledError, KeyboardInterrupt)
#: The same, and a loop over the wrapper left early, which a generator cannot tell from an exception raised in the
#: loop's body: that one closes with ``fail``.
_INTERRUPTIONS = (GeneratorExit, *_CANCELLATIONS)

# Closes scheduled for publishers handed to a refused call: the loop keeps only weak references to tasks.
_closing: set[asyncio.Task[None]] = set()


@dataclass(frozen=True)
class RunSummary:
    """What the tee knows about a run when it closes the stream."""

    #: The close line: ``end`` or ``fail``.
    outcome: Outcome
    status: RunStatus
    #: The root run's id: the ``root_run_id`` option, else the ``run_id`` of the first event of the outermost run.
    run_id: str | None
    session_id: str | None
    #: Frames this tee queued on the publisher.
    frames: int
    #: What the iterator raised, or ``before_close`` raised, if anything did.
    error: BaseException | None = None
    #: The server's answer to the close line; ``None`` in the summary given to ``before_close``.
    result: PublishResult | None = None


BeforeClose = Callable[[RunSummary], object]
"""Called after the iterator ended and before the close line; for ``apublish`` it may return an awaitable."""

ErrorHandler = Callable[[IQError], object]


@dataclass(frozen=True)
class Options:
    protocol: WireProtocol
    event_names: bool
    include: frozenset[str] | None
    exclude: frozenset[str]
    root_run_id: str | None
    close_on_cancel: Outcome
    shrink: bool
    transform: Transform | None
    expose_errors: bool
    on_dropped: ErrorHandler | None
    #: Whether the native protocol holds the root run's ending back until ``before_close`` has run.
    hold_ending: bool


def options(
    *,
    protocol: str,
    event_names: bool,
    include: Collection[str] | None,
    exclude: Collection[str],
    root_run_id: str | None,
    close_on_cancel: str,
    oversize: str,
    transform: Transform | None,
    expose_errors: bool,
    on_dropped: ErrorHandler | None,
    hold_ending: bool,
) -> Options:
    """The options, checked when ``publish()`` is called rather than at the first event."""
    if protocol not in ("agno", "ag-ui"):
        raise ValueError('protocol is "agno" or "ag-ui".')
    if close_on_cancel not in ("fail", "end"):
        raise ValueError('close_on_cancel is "fail" or "end".')
    if oversize not in ("shrink", "drop"):
        raise ValueError('oversize is "shrink" or "drop".')
    if transform is not None and protocol != "agno":
        raise ValueError("transform applies to the native protocol only.")
    if event_names and protocol != "agno":
        # StreamAgent subscribes without event names, and an EventSource hands a named frame only to its listener.
        raise ValueError("event_names applies to the native protocol only: AG-UI frames are unnamed.")
    if isinstance(include, str) or isinstance(exclude, str):
        raise ValueError("include and exclude are collections of Agno event names.")
    if protocol == "ag-ui":
        require_agui()
    return Options(
        protocol="ag-ui" if protocol == "ag-ui" else "agno",
        event_names=bool(event_names),
        include=None if include is None else frozenset(include),
        exclude=frozenset(exclude),
        root_run_id=root_run_id,
        close_on_cancel="end" if close_on_cancel == "end" else "fail",
        shrink=oversize == "shrink",
        transform=transform,
        expose_errors=bool(expose_errors),
        on_dropped=on_dropped,
        hold_ending=hold_ending,
    )


def require_agui() -> None:
    try:
        from . import _agui  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as error:
        raise ImportError(
            "The AG-UI protocol needs the ag-ui extra: pip install 'iq-stream-agno[ag-ui]'"
            f" ({error.name or error} is missing)."
        ) from error


class Reporter:
    """Adapter-level problems: to ``on_dropped``, else to the ``iq_stream_agno`` logger once per kind and run. Never
    with frame data in the message."""

    def __init__(self, handler: ErrorHandler | None) -> None:
        self._handler = handler
        self._logged: set[str | None] = set()

    def __call__(self, error: IQError) -> None:
        if self._handler is not None:
            with contextlib.suppress(Exception):
                self._handler(error)
            return
        if error.code not in self._logged:
            self._logged.add(error.code)
            logger.warning("%s", error)


class AgnoTrack:
    """The root run of an Agno iterator, and its encoder."""

    def __init__(
        self,
        options: Options,
        publisher: PublisherLike | AsyncPublisherLike,
        report: Reporter,
        agui: AguiRequest | None = None,
    ) -> None:
        self._options = options
        self._report = report
        self._stream_id = publisher.stream_id
        self._root: str | None = options.root_run_id
        # The run a nested run's first events name as theirs, until the root's own events arrive.
        self._outer: str | None = None
        self.session_id: str | None = None
        self._failed = False
        self._cancelled = False
        self._paused = False
        self._root_error: BaseRunOutputEvent | None = None
        self._unsupported = False
        #: Whether the root run's ending has passed: a pause, a completion, an error or a cancel.
        self.ended = False
        # The native frames of the root's ending, held back until before_close has run, and the event they encode.
        self._held: list[Frame] = []
        self._held_event: BaseRunOutputEvent | None = None

        def fits(data: str, event: str | None) -> bool:
            return publisher.frame_size(data, event) <= publisher.max_frame_bytes

        if options.protocol == "ag-ui":
            from ._agui import AguiEncoder, AguiFramer

            framer = AguiFramer(
                event_names=options.event_names,
                shrink=options.shrink,
                fits=fits,
                size=publisher.frame_size,
                limit=publisher.max_frame_bytes,
                report=report,
            )
            self._agui: AguiEncoder | None = AguiEncoder(framer, expose_errors=options.expose_errors, request=agui)
            self._native: NativeEncoder | None = None
        else:
            self._agui = None
            self._native = NativeEncoder(
                event_names=options.event_names,
                transform=options.transform,
                expose_errors=options.expose_errors,
                shrink=options.shrink,
                fits=fits,
                size=publisher.frame_size,
                limit=publisher.max_frame_bytes,
                report=report,
            )

    @property
    def run_id(self) -> str | None:
        return self._root or self._outer

    def frames(self, item: object, encode: bool) -> list[Frame]:
        if isinstance(item, str):
            # Agno's background mode yields SSE text instead of events.
            if not self._unsupported:
                self._unsupported = True
                self._report(
                    IQError(
                        f"IQ stream {self._stream_id}: the run yields strings (Agno's background=True), which this "
                        "adapter does not publish",
                        stream_id=self._stream_id,
                        code="unsupported_source",
                    )
                )
            return []
        if self._unsupported or not isinstance(item, BaseRunOutputEvent):
            return []  # the RunOutput that yield_run_output=True adds at the end
        name = event_name(item)
        run_id: object = getattr(item, "run_id", None)
        if self._root is None and isinstance(run_id, str) and run_id:
            # Without stream_events, a team that delegates before it writes yields its member's events first: a
            # nested run is never the root, whose own ending alone decides the close line.
            outer = nested(item)
            depth: object = getattr(item, "nested_depth", 0)
            if outer is None and not (isinstance(depth, int) and depth > 0):
                self._root = run_id
            elif self._outer is None:
                self._outer = outer
        session_id: object = getattr(item, "session_id", None)
        if self.session_id is None and isinstance(session_id, str) and session_id:
            self.session_id = session_id
        root = run_id is not None and run_id == self._root
        if root:
            if name in ROOT_ERRORS:
                self._failed = True
                self._root_error = item
            elif name in ROOT_CANCELS:
                self._cancelled = True
            elif name in ROOT_PAUSES:
                self._paused = True
            self.ended = self.ended or name in ROOT_ENDINGS
        if not encode or name in self._options.exclude:
            return []
        if self._options.include is not None and name not in self._options.include:
            return []
        try:
            if self._agui is not None:
                return self._agui.encode(item, root=root, thread_id=self.session_id, run_id=self.run_id)
            assert self._native is not None
            frames = self._native.encode(item)
        except Exception as cause:
            self._report(self._encoding_error(name, cause))
            return []
        held = self._held
        if self._options.hold_ending and root and name in ROOT_ENDINGS:
            # A page that takes RunCompleted for done must not read it before the answer is saved.
            self._held, self._held_event = frames, item
            return held
        self._held, self._held_event = [], None
        return [*held, *frames]

    def status(self) -> str:
        if self._unsupported or self._failed:
            return "error"
        if self._cancelled:
            return "cancelled"
        return "paused" if self._paused else "completed"

    def final_frames(self, status: str, error: BaseException | None, unsaved: bool) -> list[Frame]:
        """What ends the stream's data: AG-UI's terminal events; in the native protocol, the root's ending held back.
        A completion or a pause that the run turned out to fail after (the iterator raised, ``before_close`` raised) is
        replaced by an error event of its kind, and dropped when ``before_close`` was interrupted (``unsaved``: it did
        not complete); an error or a cancel goes out as it is."""
        if self._unsupported:
            return []
        if self._agui is None:
            return self._native_ending(status, error, unsaved)
        try:
            return self._agui.finish(
                status, error=error, root_error=self._root_error, thread_id=self.session_id, run_id=self.run_id
            )
        except Exception as cause:
            self._report(self._encoding_error("the end of the run", cause))
            return []

    def _native_ending(self, status: str, error: BaseException | None, unsaved: bool) -> list[Frame]:
        held, ending = self._held, self._held_event
        self._held, self._held_event = [], None
        if ending is None or self._native is None or event_name(ending) in ROOT_ERRORS | ROOT_CANCELS:
            return held  # true whether or not the answer was saved
        if status == "error":
            text = (str(error) or type(error).__name__) if error is not None and self._options.expose_errors else None
            try:
                return self._native.encode(error_event(ending, text or MASKED_ERROR))
            except Exception as cause:
                self._report(self._encoding_error("the end of the run", cause))
                return []
        return [] if unsaved else held

    def _encoding_error(self, what: str, cause: Exception) -> IQError:
        error = IQError(
            f"IQ stream {self._stream_id}: {what} could not be encoded ({type(cause).__name__})",
            stream_id=self._stream_id,
            code="encode_failed",
        )
        error.__cause__ = cause
        return error


class _Run:
    """The per-run state the two drivers share: frames queued, and the ending."""

    def __init__(
        self, publisher: PublisherLike | AsyncPublisherLike, track: AgnoTrack, close_on_cancel: Outcome
    ) -> None:
        self.publisher = publisher
        self.track = track
        self.close_on_cancel: Outcome = close_on_cancel
        self.frames = 0
        #: Called once, when the root run's ending has passed or the iterator ended, before the frames that tell the
        #: browser: a route lets a paused run be continued from then on, not only once its close line was answered.
        self.on_settled: Callable[[], object] | None = None

    def settle(self) -> None:
        callback, self.on_settled = self.on_settled, None
        if callback is not None:
            try:
                callback()
            except Exception:
                logger.exception("on_settled raised")

    def take(self, item: object) -> None:
        """Encodes one item and queues its frames. Fails open: a frame that cannot go is reported and skipped. The
        publisher's own error leaves only with the client's ``on_error="raise"``, as the application asked."""
        frames = self.track.frames(item, self.publisher.alive)
        if self.track.ended:
            self.settle()
        for event, data in frames:
            if not self._send(event, data):
                return

    def _send(self, event: str | None, data: str) -> bool:
        try:
            alive = self.publisher.send(data, event)
        except FrameTooLarge:
            return True  # measured before, so not expected; the frame is skipped
        if alive:
            self.frames += 1
        return alive

    def ending(self, error: BaseException | None) -> tuple[str, Outcome]:
        if error is not None:
            if isinstance(error, _CANCELLATIONS):
                return "interrupted", self.close_on_cancel
            return _failure_status(error), "fail"
        status = self.track.status()
        if status == "error":
            return status, "fail"
        if status == "cancelled":
            return status, self.close_on_cancel
        return status, "end"

    def finish_frames(self, status: str, error: BaseException | None, unsaved: bool) -> None:
        if self.publisher.alive:
            for event, data in self.track.final_frames(status, error, unsaved):
                if not self._send(event, data):
                    break
        self.settle()

    def summary(
        self,
        outcome: Outcome,
        status: str,
        error: BaseException | None,
        result: PublishResult | None = None,
    ) -> RunSummary:
        return RunSummary(
            outcome=outcome,
            status=_status(status),
            run_id=self.track.run_id,
            session_id=self.track.session_id,
            frames=self.frames,
            error=error,
            result=result,
        )


def _status(value: str) -> RunStatus:
    for status in ("completed", "paused", "error", "cancelled", "interrupted"):
        if value == status:
            return status
    return "error"


def _failure_status(error: BaseException) -> str:
    return "interrupted" if isinstance(error, _INTERRUPTIONS) else "error"


def _iterate(source: Iterable[E]) -> Iterator[E]:
    """The source's iterator, taken inside the driver's ``try`` so that a source that is none still closes the
    stream."""
    checked = cast("object", source)
    if isinstance(checked, Iterable):
        return iter(source)
    raise TypeError(
        "publish() takes the iterator of run(..., stream=True) or continue_run(..., stream=True); "
        f"it was given {type(source).__name__}"
    )


def _aiterate(source: AsyncIterable[E]) -> AsyncIterator[E]:
    checked = cast("object", source)
    if isinstance(checked, AsyncIterable):
        return source.__aiter__()
    if inspect.iscoroutine(checked):
        checked.close()  # arun() without stream=True: the run never starts, and no warning says it was not awaited
    raise TypeError(
        "apublish() takes the async iterator of arun(..., stream=True) or acontinue_run(..., stream=True); "
        f"it was given {type(source).__name__}"
    )


def drive(
    source: Iterable[E],
    run: _Run,  # its publisher is a PublisherLike
    before_close: BeforeClose | None,
    on_summary: Callable[[RunSummary], object] | None = None,
) -> Iterator[E]:
    publisher = cast("PublisherLike", run.publisher)
    iterator: Iterator[E] | None = None
    error: BaseException | None = None
    exhausted = False
    try:
        iterator = _iterate(source)
        for item in iterator:
            run.take(item)
            yield item
        exhausted = True
    except BaseException as caught:
        error = caught
        raise
    finally:
        if not exhausted:
            # Agno stores a run that is closed early as cancelled.
            close_source = getattr(iterator, "close", None)
            if callable(close_source):
                with contextlib.suppress(Exception):
                    close_source()
        status, outcome = run.ending(error)
        failure: BaseException | None = None
        if exhausted and before_close is not None:
            try:
                before_close(run.summary(outcome, status, None))
            except BaseException as caught:
                failure = caught
                status, outcome = run.ending(caught)
        run.finish_frames(status, error or failure, failure is not None)
        result: PublishResult | None = None
        try:
            result = publisher.close(outcome)
        except IQError:
            if error is None and failure is None:
                raise
        finally:
            if on_summary is not None:
                on_summary(run.summary(outcome, status, error or failure, result))
        if failure is not None:
            raise failure


async def adrive(
    source: Callable[[], AsyncIterable[E]],
    run: _Run,
    before_close: BeforeClose | None,
    on_summary: Callable[[RunSummary], object] | None = None,
) -> AsyncIterator[E]:
    """Drives the iterator that ``source()`` returns; ``source`` is called inside the ``try``, so that a run that
    raises when it is started (``arun()`` refusing its input) still ends its frames and closes the stream."""
    iterator: AsyncIterator[E] | None = None
    error: BaseException | None = None
    exhausted = False
    try:
        iterator = _aiterate(source())
        while True:
            try:
                item = await iterator.__anext__()
            except StopAsyncIteration:
                exhausted = True
                break
            run.take(item)
            yield item
    except BaseException as caught:
        error = caught
        raise
    finally:
        if not exhausted:
            close_source: object = getattr(iterator, "aclose", None)
            if callable(close_source):
                with contextlib.suppress(Exception):
                    await cast("Callable[[], Awaitable[object]]", close_source)()
        status, outcome = run.ending(error)
        failure: BaseException | None = None
        if exhausted and before_close is not None:
            try:
                pending = before_close(run.summary(outcome, status, None))
                if isinstance(pending, Awaitable):
                    await cast("Awaitable[object]", pending)
            except BaseException as caught:
                failure = caught
                status, outcome = run.ending(caught)
        run.finish_frames(status, error or failure, failure is not None)
        result: PublishResult | None = None
        try:
            result = await aclose(run.publisher, outcome)
        except IQError:
            if error is None and failure is None:
                raise
        finally:
            if on_summary is not None:
                on_summary(run.summary(outcome, status, error or failure, result))
        if failure is not None:
            raise failure


async def aclose(publisher: PublisherLike | AsyncPublisherLike, outcome: Outcome) -> PublishResult:
    """Closes either kind of publisher from a coroutine. The synchronous one waits for the server in a thread; both
    complete when the calling task is being cancelled."""
    if inspect.iscoroutinefunction(publisher.close):
        return await cast("AsyncPublisherLike", publisher).close(outcome)
    return await asyncio.shield(asyncio.to_thread(cast("PublisherLike", publisher).close, outcome))


def publish(
    events: Iterable[E],
    publisher: PublisherLike,
    *,
    protocol: WireProtocol = "agno",
    event_names: bool = False,
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
    root_run_id: str | None = None,
    close_on_cancel: Outcome = "fail",
    oversize: Literal["shrink", "drop"] = "shrink",
    transform: Transform | None = None,
    expose_errors: bool = False,
    on_dropped: ErrorHandler | None = None,
    before_close: Callable[[RunSummary], object] | None = None,
) -> Iterator[E]:
    """Yields every item of ``events`` (from ``Agent.run``, ``Team.run``, ``Workflow.run`` or ``continue_run`` with
    ``stream=True``) unchanged, publishes each Agno event as frames, and closes the stream when the iterator ends:
    ``end`` when the run completed or paused; ``close_on_cancel`` when Agno cancelled it or the process was
    interrupted; ``fail`` when it failed, the iterator raised or the caller stopped early. Iterate it to the end, or
    close it, so that the stream is closed. A call refused for its arguments closes the stream with ``fail``."""
    try:
        if inspect.iscoroutinefunction(publisher.close):
            raise TypeError("publish() takes a synchronous publisher; an AsyncPublisher goes with apublish().")
        checked = options(
            protocol=protocol,
            event_names=event_names,
            include=include,
            exclude=exclude,
            root_run_id=root_run_id,
            close_on_cancel=close_on_cancel,
            oversize=oversize,
            transform=transform,
            expose_errors=expose_errors,
            on_dropped=on_dropped,
            hold_ending=before_close is not None,
        )
    except BaseException:
        close_refused(publisher)
        raise
    run = _Run(publisher, AgnoTrack(checked, publisher, Reporter(on_dropped)), checked.close_on_cancel)
    return drive(events, run, before_close)


def apublish(
    events: AsyncIterable[E],
    publisher: AsyncPublisherLike | PublisherLike,
    *,
    protocol: WireProtocol = "agno",
    event_names: bool = False,
    include: Collection[str] | None = None,
    exclude: Collection[str] = (),
    root_run_id: str | None = None,
    close_on_cancel: Outcome = "fail",
    oversize: Literal["shrink", "drop"] = "shrink",
    transform: Transform | None = None,
    expose_errors: bool = False,
    on_dropped: ErrorHandler | None = None,
    before_close: Callable[[RunSummary], Awaitable[object] | object] | None = None,
) -> AsyncIterator[E]:
    """``publish()`` for ``arun`` and ``acontinue_run``. The publisher is an ``AsyncPublisher``, or a synchronous
    ``Publisher`` when async tools may block the event loop and starve the asyncio writer's keepalives. A cancelled
    task closes the stream with ``close_on_cancel``."""
    try:
        checked = options(
            protocol=protocol,
            event_names=event_names,
            include=include,
            exclude=exclude,
            root_run_id=root_run_id,
            close_on_cancel=close_on_cancel,
            oversize=oversize,
            transform=transform,
            expose_errors=expose_errors,
            on_dropped=on_dropped,
            hold_ending=before_close is not None,
        )
    except BaseException:
        close_refused(publisher)
        raise
    run = _Run(publisher, AgnoTrack(checked, publisher, Reporter(on_dropped)), checked.close_on_cancel)
    return adrive(lambda: events, run, before_close)


def close_refused(publisher: PublisherLike | AsyncPublisherLike) -> None:
    """Closes with ``fail`` the stream of a call refused for its arguments: its request is open since the publisher
    was made, and would otherwise be held open by keepalives with no close line for the page. In a running loop the
    close is scheduled, since neither kind of publisher may block the loop while the server answers."""
    try:
        loop: asyncio.AbstractEventLoop | None = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop is None:
        if not inspect.iscoroutinefunction(publisher.close):  # an AsyncPublisher without its loop cannot be closed
            with contextlib.suppress(Exception):
                cast("PublisherLike", publisher).close("fail")
        return

    async def closing() -> None:
        with contextlib.suppress(Exception):
            await aclose(publisher, "fail")

    task = loop.create_task(closing())
    _closing.add(task)
    task.add_done_callback(_closing.discard)
