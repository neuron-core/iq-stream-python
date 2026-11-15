"""``protocol="ag-ui"``: AG-UI events over CrewAI's native ones, built with the ``ag_ui.core`` models and written as
AG-UI's own encoder writes them (camelCase, members without a value left out), without an SSE event name."""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import ag_ui.core as ag
from crewai.events.base_events import BaseEvent
from pydantic import BaseModel

from ._limits import Fitter, Frame, dumps
from ._snapshot import MASKED, Snapshot, Snapshotter

try:
    from crewai.utilities.string_utils import sanitize_tool_name as _sanitize
except ImportError:  # pragma: no cover - older CrewAI

    def _sanitize(name: str) -> str:
        return name


#: Exceptions that end a block as a cancellation rather than as a failure.
CANCELLATIONS = (asyncio.CancelledError, KeyboardInterrupt)
# Step names and the agent's name on a message are labels: a task without a name is named by its whole description.
_LABEL = 120


@dataclass(frozen=True)
class Foreign:
    """An event that is already an AG-UI event, as flows written for ``ag-ui-crewai`` emit them."""

    payload: dict[str, Any]


@dataclass(eq=False)
class _ToolCall:
    """A tool call: streamed by a model, or announced when its tool started."""

    id: str
    name: str
    agent_id: str | None
    #: The arguments: streamed text, or what the tool was run with.
    args: object = ""
    ended: bool = False


@dataclass
class _Call:
    """One model call, by its ``call_id``."""

    agent_id: str | None = None
    agent_role: str | None = None
    text_open: bool = False
    reasoning_open: bool = False
    #: Whether anything of the call reached the wire, so that its answer is not published twice.
    streamed: bool = False
    #: Its streamed tool calls, by index.
    tools: dict[int, _ToolCall] = field(default_factory=lambda: {})


def _error_text(error: BaseException) -> str:
    text = str(error)
    return f"{type(error).__name__}: {text}" if text else type(error).__name__


def _label(text: str) -> str:
    return text if len(text) <= _LABEL else text[: _LABEL - 1] + "…"


def _args(value: object) -> object:
    """Arguments as a value, so that streamed JSON text compares equal to the dict a tool was run with."""
    if isinstance(value, str):
        try:
            return json.loads(value or "{}")
        except ValueError:
            return value
    return value


def _same_tool(tool_call: _ToolCall, name: str, agent_id: str | None) -> bool:
    """Whether a tool event is about this call: the same tool, by the same agent when both are known."""
    if tool_call.agent_id is not None and agent_id is not None and tool_call.agent_id != agent_id:
        return False
    return tool_call.name == name or _sanitize(tool_call.name) == _sanitize(name)


def _agui_payload(event: BaseEvent) -> dict[str, Any]:
    """The AG-UI members of an event that is also a CrewAI event, without CrewAI's own."""
    model = next(
        kind for kind in type(event).__mro__ if issubclass(kind, ag.BaseEvent) and kind.__module__.startswith("ag_ui.")
    )
    # CrewAI's members of the same name (its datetime ``timestamp``) are not AG-UI's.
    members = (set(model.model_fields) - set(BaseEvent.model_fields)) | {"type"}
    plain = model.model_validate(event.model_dump(include=members))
    return plain.model_dump(mode="json", by_alias=True, exclude_none=True)


class AgUiEncoder:
    def __init__(
        self, snapshotter: Snapshotter, fitter: Fitter, thread_id: str, run_id: str, expose_errors: bool
    ) -> None:
        self._snapshot = snapshotter
        self._fit = fitter
        self._thread_id = thread_id
        self._run_id = run_id
        self._expose_errors = expose_errors
        self._calls: dict[str, _Call] = {}
        self._awaiting: list[_ToolCall] = []  # streamed, not yet run
        self._running: list[_ToolCall] = []  # started, without a result yet
        self._ids: set[str] = set()
        self._steps: dict[tuple[str, ...], list[str]] = {}
        self._open_steps: dict[str, None] = {}  # names in the order they were opened
        # The flows paused for human feedback, by id, unless they finish afterwards: the run ends interrupted.
        self._paused: dict[str, ag.Interrupt] = {}
        self._handlers: dict[str, Callable[[Snapshot], list[Frame]]] = {
            "llm_call_started": self._call_started,
            "llm_stream_chunk": self._chunk,
            "llm_thinking_chunk": self._thinking,
            "llm_call_completed": self._call_ended,
            "llm_call_failed": self._call_ended,
            "tool_usage_started": self._tool_started,
            "tool_usage_finished": self._tool_ended,
            "tool_usage_error": self._tool_ended,
            "task_started": self._step_started,
            "task_completed": self._step_ended,
            "task_failed": self._step_ended,
            "method_execution_started": self._step_started,
            "method_execution_finished": self._step_ended,
            "method_execution_failed": self._step_ended,
            "human_feedback_requested": self._custom,
            "flow_paused": self._flow_paused,
            "flow_finished": self._flow_ended,
            "flow_failed": self._flow_ended,
        }

    # -- the Encoder protocol -------------------------------------------------------------------------------------

    def snapshot(self, source: object, event: BaseEvent) -> Snapshot | Foreign | None:
        if isinstance(event, ag.BaseEvent):
            return Foreign(_agui_payload(event))
        return self._snapshot(source, event)

    def start(self) -> list[Frame]:
        return self._out(ag.RunStartedEvent(thread_id=self._thread_id, run_id=self._run_id))

    def frames(self, snapshot: Snapshot | Foreign) -> list[Frame]:
        if isinstance(snapshot, Foreign):
            return self._fit.whole(snapshot.payload)
        if not self._snapshot.admit(snapshot):
            return []
        handler = self._handlers.get(snapshot.type)
        return [] if handler is None else handler(snapshot)

    def finish(self, error: BaseException | None) -> list[Frame]:
        out: list[Frame] = []
        for call_id, call in self._calls.items():
            out += self._close_call(call_id, call)
        for name in reversed(self._open_steps):
            out += self._out(ag.StepFinishedEvent(step_name=name))
        self._steps.clear()
        self._open_steps.clear()
        if error is None and self._paused:
            out += self._interrupted(list(self._paused.values()))
        elif error is None:
            out += self._out(ag.RunFinishedEvent(thread_id=self._thread_id, run_id=self._run_id))
        elif isinstance(error, CANCELLATIONS):
            # A stop is not a failure of the run: AG-UI has an outcome for it.
            cancelled = ag.RunFinishedCancelledOutcome()
            out += self._out(ag.RunFinishedEvent(thread_id=self._thread_id, run_id=self._run_id, outcome=cancelled))
        else:
            # A fixed code, as across the family: the exception's class names the backend's providers, so it goes
            # into the message, and only with expose_errors.
            message = _error_text(error) if self._expose_errors else MASKED
            out += self._out(ag.RunErrorEvent(message=message, code="error"), cut=("message",))
        return out

    def _interrupted(self, interrupts: list[ag.Interrupt]) -> list[Frame]:
        """``RUN_FINISHED`` with AG-UI's interrupt outcome: the run waits for an answer, and resuming is a new run.
        Without the messages when they would not fit, so that the run is never left unfinished."""
        event = ag.RunFinishedEvent(
            thread_id=self._thread_id,
            run_id=self._run_id,
            outcome=ag.RunFinishedInterruptOutcome(interrupts=interrupts),
        )
        payload = event.model_dump(mode="json", by_alias=True, exclude_none=True)
        if not self._fit.fits(dumps(payload)):
            for interrupt in payload["outcome"]["interrupts"]:
                interrupt.pop("message", None)
        return self._fit.whole(payload)

    # -- frames -----------------------------------------------------------------------------------------------------

    def _out(self, event: BaseModel, *, split: str | None = None, cut: tuple[str, ...] = ()) -> list[Frame]:
        payload = event.model_dump(mode="json", by_alias=True, exclude_none=True)
        if split is not None:
            return self._fit.split(payload, split)
        if cut:
            return self._fit.cut(payload, cut)
        return self._fit.whole(payload)

    def _unique(self, tool_call_id: str | None) -> str:
        # Some models number their tool calls from zero in every response; a reused id would merge two calls.
        if not tool_call_id or tool_call_id in self._ids:
            tool_call_id = uuid.uuid4().hex
        self._ids.add(tool_call_id)
        return tool_call_id

    def _call(self, call_id: str) -> _Call:
        call = self._calls.get(call_id)
        if call is None:
            call = self._calls[call_id] = _Call()
        return call

    def _close_reasoning(self, call_id: str, call: _Call) -> list[Frame]:
        if not call.reasoning_open:
            return []
        call.reasoning_open = False
        message_id = f"{call_id}:reasoning"
        return self._out(ag.ReasoningMessageEndEvent(message_id=message_id)) + self._out(
            ag.ReasoningEndEvent(message_id=message_id)
        )

    def _close_text(self, call_id: str, call: _Call) -> list[Frame]:
        if not call.text_open:
            return []
        call.text_open = False
        return self._out(ag.TextMessageEndEvent(message_id=call_id))

    def _close_call(self, call_id: str, call: _Call) -> list[Frame]:
        out = self._close_reasoning(call_id, call) + self._close_text(call_id, call)
        for tool_call in call.tools.values():
            out += self._end_tool_call(tool_call)
        call.tools.clear()
        return out

    def _end_tool_call(self, tool_call: _ToolCall) -> list[Frame]:
        if tool_call.ended:
            return []
        tool_call.ended = True
        return self._out(ag.ToolCallEndEvent(tool_call_id=tool_call.id))

    def _text_message(self, call_id: str, call: _Call, text: str) -> list[Frame]:
        out: list[Frame] = []
        if not call.text_open:
            call.text_open = True
            name = None if call.agent_role is None else _label(call.agent_role)
            out += self._out(ag.TextMessageStartEvent(message_id=call_id, role="assistant", name=name))
        call.streamed = True
        return out + self._out(ag.TextMessageContentEvent(message_id=call_id, delta=text), split="delta")

    def _announce(self, tool_call: _ToolCall) -> list[Frame]:
        """A tool call no model streamed: the whole of it at once."""
        out = self._out(ag.ToolCallStartEvent(tool_call_id=tool_call.id, tool_call_name=tool_call.name))
        args = tool_call.args if isinstance(tool_call.args, str) else dumps(tool_call.args or {})
        if args:
            out += self._out(ag.ToolCallArgsEvent(tool_call_id=tool_call.id, delta=args), split="delta")
        return out + self._end_tool_call(tool_call)

    # -- events -----------------------------------------------------------------------------------------------------

    def _call_started(self, snapshot: Snapshot) -> list[Frame]:
        call = self._call(snapshot.fields["call_id"])
        call.agent_id = snapshot.fields.get("agent_id")
        call.agent_role = snapshot.fields.get("agent_role")
        return []

    def _chunk(self, snapshot: Snapshot) -> list[Frame]:
        call_id: str = snapshot.fields["call_id"]
        chunk: str = snapshot.fields.get("chunk") or ""
        delta: dict[str, Any] | None = snapshot.fields.get("tool_call")
        call = self._call(call_id)
        out = self._close_reasoning(call_id, call)
        if delta is None:
            return out + self._text_message(call_id, call, chunk) if chunk else out
        tool_call = call.tools.get(delta["index"])
        if tool_call is None:
            out += self._close_text(call_id, call)
            tool_call = _ToolCall(self._unique(delta["id"]), delta["name"] or "tool", call.agent_id)
            call.tools[delta["index"]] = tool_call
            call.streamed = True
            self._awaiting.append(tool_call)
            start = ag.ToolCallStartEvent(
                tool_call_id=tool_call.id, tool_call_name=tool_call.name, parent_message_id=call_id
            )
            out += self._out(start)
        if chunk and not tool_call.ended:
            tool_call.args = f"{tool_call.args}{chunk}"
            out += self._out(ag.ToolCallArgsEvent(tool_call_id=tool_call.id, delta=chunk), split="delta")
        return out

    def _thinking(self, snapshot: Snapshot) -> list[Frame]:
        call_id: str = snapshot.fields["call_id"]
        chunk: str = snapshot.fields.get("chunk") or ""
        if not chunk:
            return []
        call = self._call(call_id)
        message_id = f"{call_id}:reasoning"
        out: list[Frame] = []
        if not call.reasoning_open:
            call.reasoning_open = True
            call.streamed = True
            out += self._out(ag.ReasoningStartEvent(message_id=message_id))
            out += self._out(ag.ReasoningMessageStartEvent(message_id=message_id, role="reasoning"))
        return out + self._out(ag.ReasoningMessageContentEvent(message_id=message_id, delta=chunk), split="delta")

    def _call_ended(self, snapshot: Snapshot) -> list[Frame]:
        call_id: str = snapshot.fields["call_id"]
        call = self._call(call_id)
        if snapshot.type == "llm_call_failed":
            # The tool calls of a failed call are not run: no later tool event may bind to them.
            self._awaiting = [tool_call for tool_call in self._awaiting if tool_call not in call.tools.values()]
        out = self._close_call(call_id, call)
        # A model without streaming announces nothing on the way: its answer is published when the call returns.
        if snapshot.response and not call.streamed:
            out += self._text_message(call_id, call, snapshot.response) + self._close_text(call_id, call)
        return out

    def _tool_started(self, snapshot: Snapshot) -> list[Frame]:
        name: str = snapshot.fields["tool_name"]
        agent_id: str | None = snapshot.fields.get("agent_id")
        args = snapshot.fields.get("tool_args")
        # Tool events carry no tool call id: bind to a streamed call of that tool by that agent, the one with the same
        # arguments if there is one, else the oldest; when no model streamed it, announce the call now.
        candidates = [tool_call for tool_call in self._awaiting if _same_tool(tool_call, name, agent_id)]
        match = next((c for c in candidates if _args(c.args) == _args(args)), candidates[0] if candidates else None)
        out: list[Frame] = []
        if match is None:
            match = _ToolCall(self._unique(None), name, agent_id, args)
            out += self._announce(match)
        else:
            self._awaiting.remove(match)
            out += self._end_tool_call(match)
        self._running.append(_ToolCall(match.id, name, agent_id, args, ended=True))
        return out

    def _tool_ended(self, snapshot: Snapshot) -> list[Frame]:
        name: str = snapshot.fields["tool_name"]
        agent_id: str | None = snapshot.fields.get("agent_id")
        args = snapshot.fields.get("tool_args")
        candidates = [tool_call for tool_call in self._running if _same_tool(tool_call, name, agent_id)]
        match = next((c for c in candidates if _args(c.args) == _args(args)), candidates[0] if candidates else None)
        out: list[Frame] = []
        if match is None:
            match = _ToolCall(self._unique(None), name, agent_id, args)
            out += self._announce(match)
        else:
            self._running.remove(match)
        if snapshot.type == "tool_usage_finished":
            content: str = snapshot.fields.get("output", "")
        else:
            content = dumps({"error": snapshot.fields.get("error", MASKED)})
        result = ag.ToolCallResultEvent(
            message_id=uuid.uuid4().hex, tool_call_id=match.id, content=content, role="tool"
        )
        return out + self._out(result, cut=("content",))

    def _step_key(self, snapshot: Snapshot) -> tuple[tuple[str, ...], str]:
        fields = snapshot.fields
        if snapshot.type.startswith("task_"):
            task_id = str(fields.get("task_id", ""))
            return ("task", task_id), f"task:{fields.get('task_name') or task_id}"
        flow, method = str(fields.get("flow_name", "")), str(fields.get("method_name", ""))
        return ("flow", flow, method), f"flow:{method}"

    def _step_started(self, snapshot: Snapshot) -> list[Frame]:
        key, label = self._step_key(snapshot)
        # Step names must be unique among the open steps of a run.
        name = base = _label(label)
        number = 2
        while name in self._open_steps:
            name = f"{base}#{number}"
            number += 1
        # A step whose STEP_FINISHED would not fit is left out, so that it is never left open.
        if not self._fit.fits(dumps({"type": "STEP_FINISHED", "stepName": name})):
            return self._fit.whole({"type": "STEP_FINISHED", "stepName": name})
        out = self._out(ag.StepStartedEvent(step_name=name))
        self._steps.setdefault(key, []).append(name)
        self._open_steps[name] = None
        return out

    def _step_ended(self, snapshot: Snapshot) -> list[Frame]:
        key, _ = self._step_key(snapshot)
        names = self._steps.get(key)
        if not names:
            return []
        name = names.pop()
        del self._open_steps[name]
        return self._out(ag.StepFinishedEvent(step_name=name))

    def _flow_paused(self, snapshot: Snapshot) -> list[Frame]:
        fields = snapshot.fields
        flow_id = fields.get("flow_id")
        if flow_id is not None:
            # The flow's id is what resumes it (Flow.from_pending).
            message = fields.get("message")
            interrupt = ag.Interrupt(id=str(flow_id), reason="human_feedback", message=message)
            self._paused[str(flow_id)] = interrupt
        return self._custom(snapshot)

    def _flow_ended(self, snapshot: Snapshot) -> list[Frame]:
        self._paused.pop(str(snapshot.fields.get("flow_id", "")), None)  # resumed, in the block
        return []

    def _custom(self, snapshot: Snapshot) -> list[Frame]:
        event = ag.CustomEvent(name=f"crewai.{snapshot.type}", value=snapshot.fields)
        payload = event.model_dump(mode="json", by_alias=True, exclude_none=True)
        return self._fit.cut(payload, ("message",), within="value")
