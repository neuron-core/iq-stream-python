"""The tee: every item of an Agno run's iterator goes on to the caller unchanged, and becomes frames for the publisher;
the stream is closed when the iterator ends, with the outcome the root run's events decide."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Collection, Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Protocol, TypeVar, cast

from agno.run.base import BaseRunOutputEvent
from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, Outcome, PublisherLike, PublishResult

from ._common import ROOT_CANCELS, ROOT_ERRORS, ROOT_PAUSES, Frame, event_name
from ._common import Protocol as WireProtocol
from ._native import NativeEncoder, Transform

if TYPE_CHECKING:
    from ._agui import AguiEncoder

E = TypeVar("E")

RunStatus = Literal["completed", "paused", "error", "cancelled", "interrupted"]
"""How a run ended: Agno completed it, paused it for a human, failed it, cancelled it; or it was ``interrupted``,
because the consumer stopped iterating or its task was cancelled."""

logger = logging.getLogger("iq_stream_agno")

_INTERRUPTIONS = (GeneratorExit, asyncio.CancelledError, KeyboardInterrupt)


@dataclass(frozen=True)
class RunSummary:
    """What the tee knows about a run when it closes the stream."""

    #: The close line: ``end`` or ``fail``.
    outcome: Outcome
    status: RunStatus
    #: The root run's id: the ``root_run_id`` option, else the ``run_id`` of the first event.
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


class _Track(Protocol):
    """A run's terminal decision and its encoder, as the drivers below see them."""

    run_id: str | None
    session_id: str | None

    def frames(self, item: object, encode: bool) -> list[Frame]: ...

    def status(self) -> str: ...

    def final_frames(self, status: str, error: BaseException | None) -> list[Frame]: ...


@dataclass(frozen=True)
class Options:
    protocol: WireProtocol
    event_names: bool
    include: frozenset[str] | None
    exclude: frozenset[str]
    root_run_id: str | None
    cancelled: Outcome
    shrink: bool
    transform: Transform | None
    expose_errors: bool
    on_error: ErrorHandler | None


def options(
    *,
    protocol: str,
    event_names: bool,
    include: Collection[str] | None,
    exclude: Collection[str],
    root_run_id: str | None,
    cancelled: str,
    on_oversize: str,
    transform: Transform | None,
    expose_errors: bool,
    on_error: ErrorHandler | None,
) -> Options:
    """The options, checked when ``publish()`` is called rather than at the first event."""
    if protocol not in ("agno", "agui"):
        raise ValueError('protocol is "agno" or "agui".')
    if cancelled not in ("fail", "end"):
        raise ValueError('cancelled is "fail" or "end".')
    if on_oversize not in ("shrink", "drop"):
        raise ValueError('on_oversize is "shrink" or "drop".')
    if transform is not None and protocol != "agno":
        raise ValueError("transform applies to the native protocol only.")
    if isinstance(include, str) or isinstance(exclude, str):
        raise ValueError("include and exclude are collections of Agno event names.")
    if protocol == "agui":
        require_agui()
    return Options(
        protocol="agui" if protocol == "agui" else "agno",
        event_names=bool(event_names),
        include=None if include is None else frozenset(include),
        exclude=frozenset(exclude),
        root_run_id=root_run_id,
        cancelled="end" if cancelled == "end" else "fail",
        shrink=on_oversize == "shrink",
        transform=transform,
        expose_errors=bool(expose_errors),
        on_error=on_error,
    )


def require_agui() -> None:
    try:
        from . import _agui  # noqa: F401  # pyright: ignore[reportUnusedImport]
    except ImportError as error:
        raise ImportError(
            "The AG-UI protocol needs the agui extra: pip install 'iq-stream-agno[agui]'"
            f" ({error.name or error} is missing)."
        ) from error


class Reporter:
    """Adapter-level problems: to ``on_error``, else to the ``iq_stream_agno`` logger once per kind and run. Never
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

    def __init__(self, options: Options, publisher: PublisherLike | AsyncPublisherLike, report: Reporter) -> None:
        self._options = options
        self._report = report
        self._stream_id = publisher.stream_id
        self.run_id: str | None = options.root_run_id
        self.session_id: str | None = None
        self._failed = False
        self._cancelled = False
        self._paused = False
        self._root_error: BaseRunOutputEvent | None = None
        self._unsupported = False

        def fits(data: str, event: str | None) -> bool:
            return publisher.frame_size(data, event) <= publisher.max_frame_bytes

        if options.protocol == "agui":
            from ._agui import AguiEncoder, AguiFramer

            framer = AguiFramer(
                event_names=options.event_names,
                mask_errors=False,  # masked on the Agno event, before the conversion
                shrink=options.shrink,
                fits=fits,
                size=publisher.frame_size,
                limit=publisher.max_frame_bytes,
                report=report,
            )
            self._agui: AguiEncoder | None = AguiEncoder(framer, expose_errors=options.expose_errors)
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
        if self.run_id is None and isinstance(run_id, str) and run_id:
            self.run_id = run_id
        session_id: object = getattr(item, "session_id", None)
        if self.session_id is None and isinstance(session_id, str) and session_id:
            self.session_id = session_id
        root = run_id is not None and run_id == self.run_id
        if root:
            if name in ROOT_ERRORS:
                self._failed = True
                self._root_error = item
            elif name in ROOT_CANCELS:
                self._cancelled = True
            elif name in ROOT_PAUSES:
                self._paused = True
        if not encode or name in self._options.exclude:
            return []
        if self._options.include is not None and name not in self._options.include:
            return []
        try:
            if self._agui is not None:
                return self._agui.encode(item, root=root, thread_id=self.session_id, run_id=self.run_id)
            assert self._native is not None
            return self._native.encode(item)
        except Exception as cause:
            self._report(self._encoding_error(name, cause))
            return []

    def status(self) -> str:
        if self._unsupported or self._failed:
            return "error"
        if self._cancelled:
            return "cancelled"
        return "paused" if self._paused else "completed"

    def final_frames(self, status: str, error: BaseException | None) -> list[Frame]:
        if self._agui is None or self._unsupported:
            return []
        try:
            return self._agui.finish(
                status, error=error, root_error=self._root_error, thread_id=self.session_id, run_id=self.run_id
            )
        except Exception as cause:
            self._report(self._encoding_error("the end of the run", cause))
            return []

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
        self, publisher: PublisherLike | AsyncPublisherLike, track: _Track, cancelled_outcome: Outcome
    ) -> None:
        self.publisher = publisher
        self.track = track
        self.cancelled_outcome: Outcome = cancelled_outcome
        self.frames = 0

    def take(self, item: object) -> None:
        """Encodes one item and queues its frames. Fails open: a frame that cannot go is reported and skipped. The
        publisher's own error leaves only with the client's ``on_error="raise"``, as the application asked."""
        for event, data in self.track.frames(item, self.publisher.alive):
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
            return _failure_status(error), "fail"
        status = self.track.status()
        if status == "error":
            return status, "fail"
        if status == "cancelled":
            return status, self.cancelled_outcome
        return status, "end"

    def finish_frames(self, status: str, error: BaseException | None) -> None:
        if self.publisher.alive:
            for event, data in self.track.final_frames(status, error):
                if not self._send(event, data):
                    return

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


def drive(
    source: Iterable[E],
    run: _Run,  # its publisher is a PublisherLike
    before_close: BeforeClose | None,
    on_summary: Callable[[RunSummary], object] | None = None,
) -> Iterator[E]:
    publisher = cast("PublisherLike", run.publisher)
    iterator = iter(source)
    error: BaseException | None = None
    exhausted = False
    try:
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
                status, outcome = _failure_status(caught), "fail"
        run.finish_frames(status, error or failure)
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
    source: AsyncIterable[E],
    run: _Run,
    before_close: BeforeClose | None,
    on_summary: Callable[[RunSummary], object] | None = None,
) -> AsyncIterator[E]:
    iterator = source.__aiter__()
    error: BaseException | None = None
    exhausted = False
    try:
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
                status, outcome = _failure_status(caught), "fail"
        run.finish_frames(status, error or failure)
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
    cancelled: Outcome = "fail",
    on_oversize: Literal["shrink", "drop"] = "shrink",
    transform: Transform | None = None,
    expose_errors: bool = False,
    on_error: ErrorHandler | None = None,
    before_close: Callable[[RunSummary], object] | None = None,
) -> Iterator[E]:
    """Yields every item of ``events`` (from ``Agent.run``, ``Team.run``, ``Workflow.run`` or ``continue_run`` with
    ``stream=True``) unchanged, publishes each Agno event as frames, and closes the stream when the iterator ends:
    ``fail`` when the root run errored or was cancelled, the iterator raised or the caller stopped early, ``end``
    otherwise. Iterate it to the end, or close it, so that the stream is closed."""
    if inspect.iscoroutinefunction(publisher.close):
        raise TypeError("publish() takes a synchronous publisher; an AsyncPublisher goes with apublish().")
    checked = options(
        protocol=protocol,
        event_names=event_names,
        include=include,
        exclude=exclude,
        root_run_id=root_run_id,
        cancelled=cancelled,
        on_oversize=on_oversize,
        transform=transform,
        expose_errors=expose_errors,
        on_error=on_error,
    )
    run = _Run(publisher, AgnoTrack(checked, publisher, Reporter(on_error)), checked.cancelled)
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
    cancelled: Outcome = "fail",
    on_oversize: Literal["shrink", "drop"] = "shrink",
    transform: Transform | None = None,
    expose_errors: bool = False,
    on_error: ErrorHandler | None = None,
    before_close: Callable[[RunSummary], Awaitable[object] | object] | None = None,
) -> AsyncIterator[E]:
    """``publish()`` for ``arun`` and ``acontinue_run``. The publisher is an ``AsyncPublisher``, or a synchronous
    ``Publisher`` when async tools may block the event loop and starve the asyncio writer's keepalives."""
    checked = options(
        protocol=protocol,
        event_names=event_names,
        include=include,
        exclude=exclude,
        root_run_id=root_run_id,
        cancelled=cancelled,
        on_oversize=on_oversize,
        transform=transform,
        expose_errors=expose_errors,
        on_error=on_error,
    )
    run = _Run(publisher, AgnoTrack(checked, publisher, Reporter(on_error)), checked.cancelled)
    return adrive(events, run, before_close)
