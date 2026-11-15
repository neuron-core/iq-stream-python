"""AG-UI as the wire protocol: relay the events of ``ag_ui_langgraph.LangGraphAgent.run()``.

Needs the ``ag-ui`` extra (``pip install "iq-stream-langchain[ag-ui]"``). Nothing of AG-UI is imported until it is
used: ``ag_ui_langgraph`` owns the run, and this module only relays what it yields.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator, Callable
from typing import TYPE_CHECKING, Any

from iq_stream import AsyncPublisherLike, FrameTooLarge, IQError, PublisherLike

from ._encode import MASKED_MESSAGE, TRUNCATED_MARKER, dumps, longest_fit
from ._relay import CancelledOutcome, aclose_publisher, stopped
from ._reports import Oversize, Reports

if TYPE_CHECKING:
    from ag_ui.core import BaseEvent, RunAgentInput
    from ag_ui_langgraph import LangGraphAgent  # pyright: ignore[reportMissingTypeStubs]

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
# Fields an event can do without, dropped when it does not fit.
_OPTIONAL = {"RUN_STARTED": "input", "RUN_FINISHED": "result"}


class _AguiRun:
    def __init__(
        self,
        pub: AsyncPublisherLike | PublisherLike,
        run_input: RunAgentInput,
        expose_errors: bool,
        on_oversize: Callable[[Oversize], None] | None,
    ) -> None:
        self.pub = pub
        self.thread_id = run_input.thread_id
        self.run_id = run_input.run_id
        self.expose_errors = expose_errors
        self.reports = Reports(pub.stream_id, on_oversize, "event")
        self.started = False
        #: RUN_FINISHED or RUN_ERROR, once relayed.
        self.terminal: str | None = None
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
        elif kind in ("RUN_FINISHED", "RUN_ERROR"):
            self.terminal = kind
        # A publisher that gave up is not fed: nothing is encoded for the rest of the run, which goes on.
        if stopped(self.pub):
            return
        try:
            fields = event.model_dump(mode="json", by_alias=True, exclude_none=True)
            if kind.endswith("_ERROR") and not self.expose_errors:
                fields = {**fields, "message": MASKED_MESSAGE}
                fields.pop("rawEvent", None)
            data = dumps(fields)
        except Exception:
            # ag_ui_langgraph passes custom values and the state as they are: one of them may not encode.
            self.reports.unencodable(kind)
            return
        self.track(kind, fields)
        try:
            self.pub.send(data)
        except FrameTooLarge as error:
            self._oversize(fields, error)

    def _fits(self, fields: dict[str, Any]) -> bool:
        return self.pub.frame_size(dumps(fields)) <= self.pub.max_frame_bytes

    def _oversize(self, fields: dict[str, Any], error: FrameTooLarge) -> None:
        kind = str(fields.get("type"))
        if kind in _DELTAS and isinstance(fields.get("delta"), str) and self._fits({**fields, "delta": "x"}):
            # Several deltas of the same id: the client concatenates them, nothing is lost.
            self._split(fields, fields["delta"], error)
            return
        reduced: dict[str, Any] | None = None
        content = fields.get("content")
        if kind == "TOOL_CALL_RESULT" and isinstance(content, str) and self._fits({**fields, "content": ""}):
            keep = longest_fit(
                len(content), lambda keep: self._fits({**fields, "content": content[:keep] + TRUNCATED_MARKER})
            )
            reduced = {**fields, "content": content[:keep] + TRUNCATED_MARKER}
        elif kind in _OPTIONAL:
            reduced = {key: value for key, value in fields.items() if key != _OPTIONAL[kind]}
            if not self._fits(reduced):
                reduced = None
        if reduced is not None:
            self.pub.send(dumps(reduced))
        self.reports.oversize(Oversize(kind, error.size, error.limit, truncated=reduced is not None))

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
        """The run raised past ``ag_ui_langgraph``: the ``RUN_ERROR`` it did not send."""
        if self.terminal is None:
            self._start()
            self.own({"type": "RUN_ERROR", "message": str(error) if self.expose_errors else MASKED_MESSAGE})

    def cancel(self) -> None:
        """A ``RUN_FINISHED`` with the ``cancelled`` outcome. AG-UI clients refuse it while a message, a tool call,
        a reasoning span, a step or a sub-agent is open, so what is open is closed first, innermost first."""
        if self.terminal is None:
            self._start()
            for _, _, closing in reversed(self.open):
                self.own(closing)
            self.own(
                {
                    "type": "RUN_FINISHED",
                    "threadId": self.thread_id,
                    "runId": self.run_id,
                    "outcome": {"type": "cancelled"},
                }
            )


def arelay_agui(
    pub: AsyncPublisherLike | PublisherLike,
    agent: LangGraphAgent,
    run_input: RunAgentInput,
    *,
    cancelled: CancelledOutcome = "fail",
    expose_errors: bool = False,
    on_oversize: Callable[[Oversize], None] | None = None,
) -> AsyncGenerator[BaseEvent, None]:
    """Runs ``agent.run(run_input)``, publishes each AG-UI event as one frame (its JSON, camelCase, as the AG-UI
    encoder writes it) and yields it unchanged.

    ``ag_ui_langgraph`` reports a failed run as a ``RUN_ERROR`` event instead of raising, so the relay closes the
    stream with ``fail`` as soon as one has been sent; the ``with`` block's own close then changes nothing. Its
    message is replaced by a fixed text unless ``expose_errors``. A cancelled run (a cancelled task, a loop left
    early under ``contextlib.aclosing()``) ends with ``RUN_FINISHED`` and a ``cancelled`` outcome, after the
    events that close what was left open, and the stream closes with ``cancelled``; for a loop left early, ``end``
    is left to the ``with`` block, as in ``relay()``. A run that finishes is left to the ``with`` block, which
    closes with ``end`` after the application has saved the answer.

    The agent is cloned for the run, as ag_ui_langgraph's own endpoint does: ``LangGraphAgent`` keeps the state of
    a run on itself, so one agent can serve concurrent requests. Construct it with ``emit_raw_events=False``: raw
    events multiply the size of a stream.
    """
    if cancelled not in ("fail", "end"):
        raise ValueError('cancelled is "fail" or "end".')
    run = _AguiRun(pub, run_input, expose_errors, on_oversize)
    return _arelay_agui(run, agent.clone(), run_input, cancelled)


async def _arelay_agui(
    run: _AguiRun, agent: LangGraphAgent, run_input: RunAgentInput, cancelled: CancelledOutcome
) -> AsyncGenerator[BaseEvent, None]:
    pub = run.pub
    events = agent.run(run_input)
    try:
        try:
            async for event in events:
                run.publish(event)
                if event.type.value == "RUN_ERROR":
                    with contextlib.suppress(IQError):
                        await aclose_publisher(pub, "fail")
                yield event
        finally:
            await events.aclose()
    except Exception as error:
        if error is not pub.error:  # with on_error="raise", the publisher's own error: nothing more can be sent
            run.fail(error)
        raise
    except BaseException as stop:  # CancelledError, GeneratorExit from aclose(), KeyboardInterrupt
        with contextlib.suppress(IQError):
            if run.terminal is None:
                run.cancel()
                # A loop left early may have been left by the caller's own exception: end is the block's to send.
                if cancelled == "fail" or not isinstance(stop, GeneratorExit):
                    await aclose_publisher(pub, cancelled)
        raise
