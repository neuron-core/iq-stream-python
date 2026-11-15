"""What is copied out of a CrewAI event, in the emitting thread, before it is encoded.

The copy is made at emit time because events hold live objects (agents, tasks, flow state) that change afterwards,
and it takes only the members listed here: CrewAI's own serialized frame of one token is about 900 bytes, and of
``agent_execution_started`` in a one-agent crew about 11 KB.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple

from crewai.events.base_events import BaseEvent

from ._limits import cut_text, dumps


def _scope_events() -> tuple[frozenset[str], frozenset[str]]:
    """The event types that open a scope in CrewAI's event context, and those that close one."""
    try:
        from crewai.events.event_context import SCOPE_ENDING_EVENTS, SCOPE_STARTING_EVENTS
    except ImportError:  # pragma: no cover - a CrewAI that moved them
        opening = {"crew_kickoff_started", "task_started", "agent_execution_started", "lite_agent_execution_started"}
        return frozenset({*opening, "llm_call_started", "tool_usage_started", "flow_started"}), frozenset()
    return frozenset(SCOPE_STARTING_EVENTS), frozenset(SCOPE_ENDING_EVENTS)


_OPENS, _ENDS = _scope_events()

MASKED = "An error occurred."

#: The members each CrewAI event type publishes with ``protocol="crewai"``. Its keys are the event types the
#: adapter knows, and the default set of ``include``.
EVENT_FIELDS: Mapping[str, tuple[str, ...]] = {
    "crew_kickoff_started": ("crew_name",),
    "crew_kickoff_completed": ("crew_name", "total_tokens", "output"),
    "crew_kickoff_failed": ("crew_name", "error"),
    "task_started": ("task_id", "task_name"),
    "task_completed": ("task_id", "task_name", "output"),
    "task_failed": ("task_id", "task_name", "error"),
    "agent_execution_started": ("agent_id", "agent_role", "task_id"),
    "agent_execution_completed": ("agent_id", "agent_role", "task_id", "output"),
    "agent_execution_error": ("agent_id", "agent_role", "task_id", "error"),
    "llm_call_started": ("call_id", "model", "agent_id", "agent_role", "task_id"),
    "llm_stream_chunk": ("call_id", "chunk"),
    "llm_thinking_chunk": ("call_id", "chunk"),
    "llm_call_completed": ("call_id", "finish_reason", "usage"),
    "llm_call_failed": ("call_id", "error"),
    "tool_usage_started": ("tool_name", "tool_args", "agent_id", "agent_role", "task_id"),
    "tool_usage_finished": ("tool_name", "tool_args", "agent_id", "task_id", "output", "from_cache"),
    "tool_usage_error": ("tool_name", "tool_args", "agent_id", "task_id", "error"),
    "flow_started": ("flow_name", "flow_id"),
    "flow_finished": ("flow_name", "flow_id"),
    "flow_failed": ("flow_name", "flow_id", "error"),
    "flow_paused": ("flow_name", "flow_id", "method_name", "message", "emit"),
    "method_execution_started": ("flow_name", "method_name"),
    "method_execution_finished": ("flow_name", "method_name"),
    "method_execution_failed": ("flow_name", "method_name", "error"),
    "human_feedback_requested": ("flow_name", "method_name", "message", "emit", "request_id"),
    "human_feedback_received": ("flow_name", "method_name", "feedback", "outcome", "request_id"),
}

# Published as text, and cut to max_field_bytes: outputs are arbitrary objects, and a task's name is its whole
# description when it has no name.
_TEXT = frozenset({"output", "error", "message", "feedback", "task_name"})
# JSON values copied, so that later changes to the event's objects do not reach the wire.
_JSON = frozenset({"tool_args", "usage", "emit"})
# Agent execution events carry the agent and the task, not their ids.
_OWNED = {"agent_id": ("agent", "id"), "agent_role": ("agent", "role"), "task_id": ("task", "id")}


#: What a scope is to ``Snapshotter.admit``: CrewAI's own from top to bottom, or CrewAI's work around an agent's run.
Role = Literal["own", "work"]


class Scope(NamedTuple):
    """Where an event was emitted, in CrewAI's tree of scopes."""

    event_id: str
    #: The scope the event was emitted in: the ``event_id`` of the event that opened it.
    parent: str | None
    #: For an event that closes a scope, the event that opened it.
    started: str | None
    #: Whether the event opens a scope.
    opens: bool
    #: What the scope it opens is, if anything to ``admit``.
    role: Role | None
    #: A model event that names no agent.
    agentless: bool
    #: For CrewAI's summary of a full context window: True as it starts, False as it ends.
    summary: bool | None = None


@dataclass(frozen=True)
class Snapshot:
    """The part of one CrewAI event the built-in encoders need."""

    type: str
    #: The event's ``emission_sequence``.
    seq: int | None
    #: The published members, without ``None`` values.
    fields: dict[str, Any] = field(default_factory=lambda: {})
    #: The answer of a model call that returned plain text, for a call that streamed nothing.
    response: str | None = None
    #: Where the event was emitted, for ``Snapshotter.admit``.
    scope: Scope | None = None
    #: False for an event that only opens or closes a scope.
    publish: bool = True


def resolve_types(include: Collection[str] | None, exclude: Collection[str]) -> frozenset[str]:
    """The event types to publish, refusing names the adapter does not know."""
    for name in (*(include or ()), *exclude):
        if name not in EVENT_FIELDS:
            raise ValueError(f"Unknown CrewAI event type {name!r}; the known ones are iq_stream_crewai.EVENT_TYPES.")
    chosen = frozenset(EVENT_FIELDS) if include is None else frozenset(include)
    return chosen - frozenset(exclude)


def as_text(value: object) -> str:
    """Tool results and task outputs are arbitrary objects; the wire wants a string."""
    if isinstance(value, str):
        return value
    raw = getattr(value, "raw", None)  # TaskOutput, CrewOutput
    if isinstance(raw, str):
        return raw
    if isinstance(value, BaseException):
        return str(value)
    try:
        return dumps(value)
    except (TypeError, ValueError):
        return str(value)


def _copy_json(value: object) -> object:
    if isinstance(value, str | int | float | bool) or value is None:
        return value
    return json.loads(dumps(value))


#: Scopes whose whole content is CrewAI's own, its agents' calls included: a task's guardrail, an agent's reasoning
#: and evaluation, memory, knowledge queries.
_OWN = frozenset(
    {"llm_guardrail_started", "agent_reasoning_started", "agent_evaluation_started"}
    | {"memory_save_started", "memory_query_started", "memory_retrieval_started"}
    | {"knowledge_query_started", "knowledge_search_query_started"}
)
#: Scopes in which a model call directly inside that names no agent is CrewAI's own: the conversion of a task's or an
#: agent's output, memory's extraction after an agent's run. The agent's own loop, a flow inside, is not.
_WORK = frozenset({"task_started", "agent_execution_started", "lite_agent_execution_started"})
#: CrewAI's summary of a full context window: model calls with no agent, in the agent's loop, between these.
_SUMMARY = {
    "context_window_summarization_started": True,
    "context_window_summarization_completed": False,
    "context_window_summarization_failed": False,
}
_STARTS = _OPENS | _OWN | _WORK
_LLM = frozenset(kind for kind in EVENT_FIELDS if kind.startswith("llm_"))
_INTERNAL_FLOWS = frozenset({"AgentExecutor", "RecallFlow", "EncodingFlow"})

#: The event types the adapter follows, published or not, to know where an event was emitted.
SCOPE_TYPES: frozenset[str] = _STARTS | _ENDS | _SUMMARY.keys()


def _is_internal_flow(source: object) -> bool:
    # The agent executor and memory's recall and encoding are Flows that emit flow_started / flow_finished; those are
    # not the application's flows. Recent CrewAI marks them; older releases are known by name.
    kind = type(source)
    if getattr(kind, "is_crewai_internal", False):
        return True
    return kind.__name__ in _INTERNAL_FLOWS and kind.__module__.startswith("crewai.")


def _is_agent_loop(source: object) -> bool:
    """The flow an agent runs its model and its tools in: what happens there is the agent's."""
    kind = type(source)
    return kind.__name__ == "AgentExecutor" and kind.__module__.startswith("crewai.")


def _is_crewai_task(source: object) -> bool:
    """A task CrewAI makes for itself (``Crew(planning=True)``'s plan, ``crew.test()``'s evaluation), known by its
    output model, which is CrewAI's."""
    model: object = getattr(source, "output_pydantic", None)
    return isinstance(model, type) and model.__module__.startswith("crewai.")


def _role(source: object, kind: str) -> Role | None:
    if kind in _OWN:
        return "own"
    if kind == "flow_started":
        return "own" if _is_internal_flow(source) and not _is_agent_loop(source) else None
    if kind == "task_started" and _is_crewai_task(source):
        return "own"
    return "work" if kind in _WORK else None


class Snapshotter:
    """Turns an event into a ``Snapshot``: cheap, and it never keeps a reference to the event. ``admit`` then says,
    in the order the events are delivered, which ones are published."""

    def __init__(self, types: frozenset[str], max_field_bytes: int | None, expose_errors: bool) -> None:
        self.types = types
        self._max_field_bytes = max_field_bytes
        self._expose_errors = expose_errors
        # The open scopes admit cares about, by the event_id of the event that opened them.
        self._open: dict[str, Role] = {}
        # The scopes in which CrewAI is summarizing a full context window.
        self._summarizing: set[str] = set()

    def admit(self, snapshot: Snapshot) -> bool:
        """Whether to publish the event, following the scopes it opens and closes. Called under the stream's lock, in
        delivery order, where a scope's opening event comes before anything emitted inside it.

        CrewAI's own is left out on positive evidence: whatever is inside a scope that is CrewAI's own, and a model
        call that names no agent directly inside a task or an agent's run (outside the agent's loop) or while CrewAI
        summarizes the context. Anything else is published, whoever made it."""
        scope = snapshot.scope
        if scope is None:
            return snapshot.publish
        around = self._open.get(scope.parent) if scope.parent is not None else None
        hidden = around == "own" or (scope.agentless and (around == "work" or scope.parent in self._summarizing))
        if scope.opens:
            # A hidden scope hides what is emitted inside it: the chunks of a hidden model call.
            role: Role | None = "own" if hidden else scope.role
            if role is not None:
                self._open[scope.event_id] = role
            hidden = role == "own"
        elif scope.started is not None:
            hidden = self._open.pop(scope.started, None) == "own" or hidden
        if scope.summary is not None and scope.parent is not None:
            if scope.summary:
                self._summarizing.add(scope.parent)
            else:
                self._summarizing.discard(scope.parent)
        return snapshot.publish and not hidden

    def _scope(self, source: object, event: BaseEvent, kind: str) -> Scope:
        parent: object = getattr(event, "parent_event_id", None)
        started: object = getattr(event, "started_event_id", None)
        return Scope(
            event.event_id,
            parent if isinstance(parent, str) else None,
            started if isinstance(started, str) else None,
            kind in _STARTS,
            _role(source, kind),
            kind in _LLM and getattr(event, "agent_id", None) is None,
            _SUMMARY.get(kind),
        )

    def __call__(self, source: object, event: BaseEvent) -> Snapshot | None:
        kind: object = getattr(event, "type", None)
        if not isinstance(kind, str):
            return None
        tracked = kind in SCOPE_TYPES
        internal = kind.startswith(("flow_", "method_execution_")) and _is_internal_flow(source)
        if kind not in self.types or internal:
            # Not published, but it may open or close a scope.
            return Snapshot(kind, None, scope=self._scope(source, event, kind), publish=False) if tracked else None
        scope = self._scope(source, event, kind) if tracked or kind in _LLM else None
        fields: dict[str, Any] = {}
        for name in EVENT_FIELDS[kind]:
            value: object = getattr(event, name, None)
            if value is None and name in _OWNED:
                owner, attribute = _OWNED[name]
                value = getattr(getattr(event, owner, None), attribute, None)
                value = None if value is None else str(value)
            elif value is None and name == "flow_id":
                value = getattr(source, "flow_id", None) or None  # the flow's own events carry only its name
            if value is None:
                continue
            if name == "error" and not self._expose_errors:
                value = MASKED
            elif name in _TEXT:
                value = cut_text(as_text(value), self._max_field_bytes)
            elif name in _JSON:
                value = _copy_json(value)
            fields[name] = value
        response: str | None = None
        if kind == "llm_stream_chunk":
            call: Any = getattr(event, "tool_call", None)
            if call is not None:
                # function.arguments is the accumulated string on every chunk; only chunk is the delta. Publishing
                # the accumulated value would make the stream grow quadratically.
                fields["tool_call"] = {"id": call.id, "name": call.function.name, "index": call.index}
        elif kind == "llm_call_completed":
            answer: object = getattr(event, "response", None)
            call_type: object = getattr(getattr(event, "call_type", None), "value", None)
            # A tool call's completion carries the tool calls, or the result of a tool the model ran itself.
            if isinstance(answer, str) and call_type == "llm_call":
                response = answer
        return Snapshot(kind, event.emission_sequence, fields, response, scope)
