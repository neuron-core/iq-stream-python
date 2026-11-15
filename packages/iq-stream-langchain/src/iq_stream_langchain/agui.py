"""AG-UI as the wire protocol: relay the events of ``ag_ui_langgraph.LangGraphAgent.run()``.

Needs the ``ag-ui`` extra (``pip install "iq-stream-langchain[ag-ui]"``). Nothing of AG-UI is imported until it is
used: ``ag_ui_langgraph`` owns the run, and this module only relays what it yields.
"""

from __future__ import annotations

import contextlib
import inspect
import json
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import TYPE_CHECKING, Any, cast

from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, Outcome, PublisherLike

from ._encode import MASKED_MESSAGE, TRUNCATED_MARKER, cut_to_fit, dumps, longest_fit
from ._relay import Feed, aclose_publisher, cancelling, check_close_on_cancel, stopped
from ._reports import Oversize, Reports

if TYPE_CHECKING:
    from ag_ui.core import BaseEvent, RunAgentInput, RunFinishedEvent
    from ag_ui_langgraph import LangGraphAgent  # pyright: ignore[reportMissingTypeStubs]

#: Called with the run's ``RUN_FINISHED`` before it is published; may be a coroutine function.
BeforeClose = Callable[["RunFinishedEvent"], "Awaitable[object] | object"]

_CANCELLED = "run cancelled"
# An event that opens something the client tracks: the events that close it, the field that names it, and what else
# the closing event the relay sends on a cancel needs.
_OPENERS: dict[str, tuple[tuple[str, ...], str, dict[str, str]]] = {
    "TEXT_MESSAGE_START": (("TEXT_MESSAGE_END",), "messageId", {}),
    "TOOL_CALL_START": (("TOOL_CALL_END",), "toolCallId", {}),
    "REASONING_START": (("REASONING_END",), "messageId", {}),
    "REASONING_MESSAGE_START": (("REASONING_MESSAGE_END",), "messageId", {}),
    "STEP_STARTED": (("STEP_FINISHED",), "stepName", {}),
    # A sub-agent of LangGraphAgent(subagent_visibility="attributed"): ended by an error on a cancel, since it did
    # not finish.
    "SUBAGENT_STARTED": (
        ("SUBAGENT_ERROR", "SUBAGENT_FINISHED"),
        "subagentRunId",
        {"message": _CANCELLED, "code": "cancelled"},
    ),
}
_CLOSERS = {closer for closers, _, _ in _OPENERS.values() for closer in closers}
# Events whose text may be sent as several events of the same id when one frame cannot hold it.
_DELTAS = {"TEXT_MESSAGE_CONTENT", "TOOL_CALL_ARGS", "REASONING_MESSAGE_CONTENT"}
# Fields an event can do without, dropped when it does not fit: a raw copy of what it carries, the run's input
# and result.
_OPTIONAL = {"RUN_STARTED": "input", "RUN_FINISHED": "result"}
# ag_ui_langgraph's event for a run that waits for a human (its default way of saying so).
_ON_INTERRUPT = "on_interrupt"


def _failed_tool_result(message: object) -> dict[str, Any] | None:
    """A tool message of a snapshot whose content is a failed tool's error text: AG-UI's (``role`` tool, ``error``
    set) or LangChain's, in a state snapshot (``type`` tool, ``status`` error)."""
    entry = cast("dict[str, Any]", message) if isinstance(message, dict) else None
    if entry is None:
        return None
    if (entry.get("role") == "tool" and entry.get("error")) or (
        entry.get("type") == "tool" and entry.get("status") == "error"
    ):
        return entry
    return None


def _masked_tools(value: object, depth: int = 3) -> object:
    """A snapshot (plain JSON) with the content of each failed tool's message replaced."""
    failed = _failed_tool_result(value)
    if failed is not None:
        return {**failed, "content": MASKED_MESSAGE}
    if depth and isinstance(value, dict):
        return {key: _masked_tools(item, depth - 1) for key, item in cast("dict[str, object]", value).items()}
    if depth and isinstance(value, list):
        return [_masked_tools(item, depth - 1) for item in cast("list[object]", value)]
    return value


class _AguiRun:
    def __init__(
        self,
        pub: AsyncPublisherLike | PublisherLike,
        run_input: RunAgentInput,
        expose_errors: bool,
        close_on_cancel: Outcome,
        on_oversize: Callable[[Oversize], None] | None,
    ) -> None:
        self.pub = pub
        self.thread_id = run_input.thread_id
        self.run_id = run_input.run_id
        self.expose_errors = expose_errors
        self.close_on_cancel: Outcome = close_on_cancel
        self.reports = Reports(pub.stream_id, on_oversize, "event")
        self.started = False
        #: The run's RUN_FINISHED, held back until the application has saved its answer.
        self.finished: RunFinishedEvent | None = None
        #: RUN_FINISHED or RUN_ERROR, once relayed.
        self.terminal: str | None = None
        #: Whether the relay ended the run as cancelled.
        self.cancelled = False
        #: What the client holds open, in the order it was opened: the events that close it, the fields that name
        #: it, and the event that closes it on a cancel.
        self.open: list[tuple[tuple[str, ...], dict[str, Any], dict[str, Any]]] = []

    def track(self, kind: str, event: dict[str, Any]) -> None:
        if kind in _OPENERS:
            closers, key, extra = _OPENERS[kind]
            names = {key: event.get(key)}
            if "subagentRunId" in event:  # attributed to a sub-agent: a step name repeats in each
                names["subagentRunId"] = event["subagentRunId"]
            self.open.append((closers, names, {"type": closers[0], **names, **extra}))
        elif kind in _CLOSERS:
            for index in range(len(self.open) - 1, -1, -1):
                closers, names, _ = self.open[index]
                if kind in closers and all(event.get(name) == value for name, value in names.items()):
                    del self.open[index]
                    break

    def publish(self, event: BaseEvent) -> None:
        kind = str(event.type.value)
        if kind == "RUN_STARTED":
            self.started = True
        elif kind == "RUN_FINISHED":
            # Nothing may tell the browser the run is over before the answer is saved: see finish().
            self.finished = cast("RunFinishedEvent", event)
            return
        elif kind == "RUN_ERROR":
            self.terminal = kind
        self._send_event(kind, event)

    def _send_event(self, kind: str, event: BaseEvent) -> None:
        # A publisher that gave up is not fed: nothing is encoded for the rest of the run, which goes on.
        if stopped(self.pub):
            return
        try:
            fields = event.model_dump(mode="json", by_alias=True, exclude_none=True)
            if not self.expose_errors:
                if kind.endswith("_ERROR"):
                    fields = {**fields, "message": MASKED_MESSAGE}
                    fields.pop("rawEvent", None)
                elif kind in ("MESSAGES_SNAPSHOT", "STATE_SNAPSHOT"):
                    fields = cast("dict[str, Any]", _masked_tools(fields))
            data = dumps(fields)
        except Exception:
            # ag_ui_langgraph passes custom values and the state as they are: one of them may not encode.
            self.reports.unencodable(kind)
            return
        self.track(kind, fields)
        try:
            self.pub.send(data)
        except FrameTooLarge as error:
            self._oversize(kind, fields, error)

    def _fits(self, fields: dict[str, Any]) -> bool:
        return self.pub.frame_size(dumps(fields)) <= self.pub.max_frame_bytes

    def _oversize(self, kind: str, fields: dict[str, Any], error: FrameTooLarge) -> None:
        if kind in _DELTAS and isinstance(fields.get("delta"), str) and self._fits({**fields, "delta": "x"}):
            # Several deltas of the same id: the client concatenates them, nothing is lost.
            self._split(fields, fields["delta"], error)
            return
        reduced = self._reduced(kind, fields)
        if reduced is not None:
            self.pub.send(dumps(reduced))
        self.reports.oversize(Oversize(kind, error.size, error.limit, truncated=reduced is not None))

    def _reduced(self, kind: str, fields: dict[str, Any]) -> dict[str, Any] | None:
        """The event cut down to fit, or ``None`` when it is not sent. The events that end the run, and the one that
        says it waits for a human, are cut as far as it takes: without them the page shows an answer that never
        ends, or nothing to approve."""
        lean = {key: value for key, value in fields.items() if key not in ("rawEvent", _OPTIONAL.get(kind))}
        if self._fits(lean):
            return lean
        content = lean.get("content")
        if kind == "TOOL_CALL_RESULT" and isinstance(content, str) and self._fits({**lean, "content": ""}):
            keep = longest_fit(
                len(content), lambda keep: self._fits({**lean, "content": content[:keep] + TRUNCATED_MARKER})
            )
            return {**lean, "content": content[:keep] + TRUNCATED_MARKER}
        if kind.endswith("_ERROR"):
            # An exposed message that does not fit goes masked, as the relay's own error part does.
            masked = {**lean, "message": MASKED_MESSAGE}
            if self._fits(masked):
                return masked
            bare = {key: lean[key] for key in ("type", "subagentRunId", "code") if key in lean}
            bare["message"] = MASKED_MESSAGE
            return bare if self._fits(bare) else None
        if kind == "CUSTOM" and lean.get("name") == _ON_INTERRUPT:
            return self._interrupt_event(lean)
        if kind == "RUN_FINISHED":
            return self._run_finished(lean)
        return None

    def _interrupt_event(self, lean: dict[str, Any]) -> dict[str, Any] | None:
        """ag_ui_langgraph writes the interrupt's value as JSON text: cut its long strings, keep its structure."""
        value = lean.get("value")
        as_text = isinstance(value, str)
        try:
            plain = json.loads(value) if isinstance(value, str) else value
        except ValueError:
            plain, as_text = value, False  # a plain string, cut as one
        cut = cut_to_fit(plain, lambda reduced: self._fits({**lean, "value": dumps(reduced) if as_text else reduced}))
        if cut is None:
            return None
        return {**lean, "value": dumps(cut) if as_text else cut}

    def _run_finished(self, lean: dict[str, Any]) -> dict[str, Any] | None:
        """An interrupt outcome keeps each interrupt's ids and the structure of what it asks, its long strings cut;
        then only the ids and reasons; then the event goes without the outcome, so that the run still ends."""
        outcome = cast("dict[str, Any]", lean["outcome"]) if isinstance(lean.get("outcome"), dict) else {}
        if outcome.get("type") == "interrupt":
            cut = cut_to_fit(outcome, lambda reduced: self._fits({**lean, "outcome": reduced}))
            if cut is not None:
                return {**lean, "outcome": cut}
            named: list[dict[str, Any]] = []
            for item in cast("list[object]", outcome.get("interrupts") or []):
                fields = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
                named.append(
                    {key: fields[key] for key in ("id", "reason", "toolCallId", "subagentRunId") if key in fields}
                )
            minimal = {**lean, "outcome": {"type": "interrupt", "interrupts": named}}
            if named and self._fits(minimal):
                return minimal
        bare = {key: lean[key] for key in ("type", "threadId", "runId") if key in lean}
        return bare if self._fits(bare) else None

    def _split(self, fields: dict[str, Any], text: str, error: FrameTooLarge) -> None:
        while text:
            keep = longest_fit(len(text), lambda keep: self._fits({**fields, "delta": text[:keep]}))  # noqa: B023
            if keep == 0:  # a single character escaped beyond the limit
                self.reports.oversize(Oversize(str(fields["type"]), error.size, error.limit, truncated=True))
                return
            self.pub.send(dumps({**fields, "delta": text[:keep]}))
            text = text[keep:]

    def own(self, fields: dict[str, Any]) -> None:
        """Sends an event the adapter made, if the publisher still takes frames."""
        if self.pub.alive:
            with contextlib.suppress(FrameTooLarge):
                self.pub.send(dumps(fields))

    def _start(self) -> None:
        if not self.started:
            self.own({"type": "RUN_STARTED", "threadId": self.thread_id, "runId": self.run_id})

    def fail(self, error: BaseException) -> None:
        """The run raised past ``ag_ui_langgraph``, or the application's save did: the ``RUN_ERROR`` that says so."""
        if self.terminal is not None or error is self.pub.error:
            return  # with on_error="raise", the publisher's own error: nothing more can be sent
        self.terminal = "RUN_ERROR"
        self._start()
        if self.pub.alive:
            with contextlib.suppress(FrameTooLarge):
                try:
                    message = str(error) if self.expose_errors else MASKED_MESSAGE
                    self.pub.send(dumps({"type": "RUN_ERROR", "message": message}))
                except FrameTooLarge:
                    self.pub.send(dumps({"type": "RUN_ERROR", "message": MASKED_MESSAGE}))

    def cancel(self) -> None:
        """A ``RUN_FINISHED`` with the ``cancelled`` outcome. AG-UI clients refuse it while a message, a tool call,
        a reasoning span, a step or a sub-agent is open, so what is open is closed first, innermost first."""
        if self.terminal is not None:
            return
        self.terminal = "RUN_FINISHED"
        self.cancelled = True
        self._start()
        for _, _, closing in reversed(self.open):
            self.own(closing)
        self.own(
            {"type": "RUN_FINISHED", "threadId": self.thread_id, "runId": self.run_id, "outcome": {"type": "cancelled"}}
        )

    async def astop(self) -> None:
        """The run is being stopped, before the graph is: the events of a cancelled run, and the close line at once
        when it is ``fail``."""
        with contextlib.suppress(IQError):
            self.cancel()
            if self.cancelled and self.close_on_cancel == "fail":
                await aclose_publisher(self.pub, "fail")

    async def astopped(self, *, cancelled: bool) -> None:
        """The graph has stopped: ``end`` for a cancel, when asked. A loop left by a ``break`` or by the caller's own
        exception leaves it to the caller's block, which closes ``fail`` after an exception."""
        if self.cancelled and self.close_on_cancel == "end" and (cancelled or cancelling()):
            with contextlib.suppress(IQError):
                await aclose_publisher(self.pub, "end")

    async def finish(self, before_close: BeforeClose | None) -> None:
        """The run's events are over: ``before_close`` saves the answer, then ``RUN_FINISHED`` goes out. If the save
        raises, a ``RUN_ERROR`` goes out instead and the stream closes with ``fail``."""
        finished = self.finished
        if finished is None or self.terminal is not None:
            return
        if before_close is not None:
            try:
                pending = before_close(finished)
                if inspect.isawaitable(pending):
                    await pending
            except Exception as error:
                with contextlib.suppress(IQError):
                    self.fail(error)
                    await aclose_publisher(self.pub, "fail")
                raise
            except BaseException:  # a cancel during the save
                with contextlib.suppress(IQError):
                    self.cancel()
                    await aclose_publisher(self.pub, self.close_on_cancel)
                raise
        self.terminal = "RUN_FINISHED"
        self._send_event("RUN_FINISHED", finished)


def arelay_agui(
    pub: AsyncPublisherLike | PublisherLike,
    agent: LangGraphAgent,
    run_input: RunAgentInput,
    *,
    close_on_cancel: Outcome = "fail",
    expose_errors: bool = False,
    on_oversize: Callable[[Oversize], None] | None = None,
    before_close: BeforeClose | None = None,
) -> AsyncGenerator[BaseEvent, None]:
    """Runs ``agent.run(run_input)``, publishes each AG-UI event as one frame (its JSON, camelCase, as the AG-UI
    encoder writes it) and yields it unchanged.

    The run's ``RUN_FINISHED`` is yielded when it comes and published when the run's events are over, after
    ``before_close(event)`` returned (awaited if it is a coroutine function): save the answer there, so that no
    browser is told the run finished before a late one can load it from you. If ``before_close`` raises, the page
    gets a ``RUN_ERROR`` instead, the stream closes with ``fail`` and the exception goes on.

    ``ag_ui_langgraph`` reports a failed run as a ``RUN_ERROR`` event instead of raising, so the relay closes the
    stream with ``fail`` as soon as one has been sent; the ``with`` block's own close then changes nothing. Its
    message is replaced by a fixed text unless ``expose_errors``. A cancelled run (a cancelled task, a loop left
    early under ``contextlib.aclosing()``, also at ``RUN_FINISHED``) ends with ``RUN_FINISHED`` and a ``cancelled``
    outcome, after the events that close what was left open, and the stream closes with ``close_on_cancel``; for a
    loop left early, ``end`` is left to the ``with`` block, as in ``relay()``.

    The agent is cloned for the run, as ag_ui_langgraph's own endpoint does: ``LangGraphAgent`` keeps the state of
    a run on itself, so one agent can serve concurrent requests. Construct it with ``emit_raw_events=False``: raw
    events multiply the size of a stream.
    """
    check_close_on_cancel(close_on_cancel)
    run = _AguiRun(pub, run_input, expose_errors, close_on_cancel, on_oversize)
    return _arelay_agui(run, agent.clone(), run_input, before_close)


async def _arelay_agui(
    run: _AguiRun, agent: LangGraphAgent, run_input: RunAgentInput, before_close: BeforeClose | None
) -> AsyncGenerator[BaseEvent, None]:
    pub = run.pub
    feed = Feed(agent.run(run_input))
    try:
        async for event in feed:
            run.publish(event)
            if event.type.value == "RUN_ERROR":
                with contextlib.suppress(IQError):
                    await aclose_publisher(pub, "fail")
            try:
                yield event
            except GeneratorExit:  # aclose(): the loop was left, the caller's code in it raised, or its task cancelled
                await run.astop()
                await feed.aclose()
                await run.astopped(cancelled=False)
                raise
    except GeneratorExit:
        raise
    except Exception as error:
        run.fail(error)
        raise
    except BaseException:  # CancelledError, in the wait for the next event; KeyboardInterrupt
        await run.astop()
        await feed.aclose()
        await run.astopped(cancelled=True)
        raise
    finally:
        await feed.aclose()
    await run.finish(before_close)
