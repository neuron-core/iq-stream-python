"""``NativeEncoder``: the SSE event name is the event's class name, the data a JSON projection of the event."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any

from workflows.events import (
    Event,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)

from ._types import agent_events, agui_events, failure, tool_kwargs
from .base import (
    DEFAULT_MAX_FIELD_BYTES,
    MASKED_ERROR,
    Fits,
    Frame,
    bounded_json,
    cut,
    dumps,
    event_name,
    fields,
    shrink_strings,
    split_text,
)

if TYPE_CHECKING:
    from llama_index.core.agent.workflow.workflow_events import AgentOutput, ToolCallResult

_TRUNCATED = "…"
# Never cut: what a page keys on.
_KEEP = frozenset({"tool_id", "tool_name", "agent", "run_id", "step_name", "type", "is_error", "return_direct"})


class NativeEncoder:
    """Publishes each event under its class name, as JSON a page parses itself.

    Prompts and chat history never leave the server: ``AgentInput`` and ``AgentSetup`` are not published, and
    ``AgentStream`` carries its delta, not the cumulative ``response`` nor the provider's ``raw`` chunk. Tool arguments,
    tool results and final answers longer than ``max_field_bytes`` are cut, with ``"truncated": true`` in the frame.
    ``include`` and ``exclude`` choose events by class (``isinstance``); the terminal event is always published. An
    exception's text is masked as ``"An error occurred."`` unless ``expose_errors`` is true.
    """

    def __init__(
        self,
        *,
        include: Sequence[type[Event]] = (),
        exclude: Sequence[type[Event]] = (),
        max_field_bytes: int = DEFAULT_MAX_FIELD_BYTES,
        expose_errors: bool = False,
    ) -> None:
        if max_field_bytes < 1:
            raise ValueError("max_field_bytes must be positive")
        self._include = tuple(include)
        self._exclude = tuple(exclude)
        self._max_field_bytes = max_field_bytes
        self._expose_errors = expose_errors
        self._tools: set[str] = set()  # tool ids announced: a human-input wait replays the tool step

    def start(self, run_id: str) -> Iterable[Frame]:
        return [("RunStarted", dumps({"run_id": run_id}))]

    def encode(self, event: Event) -> Iterable[Frame]:
        if isinstance(event, StopEvent):
            return [(event_name(event), dumps(self._terminal(event)))]
        if self._include and not isinstance(event, self._include):
            return []
        if self._exclude and isinstance(event, self._exclude):
            return []
        agents = agent_events()
        if agents is not None:
            if isinstance(event, (agents.AgentInput, agents.AgentSetup)):
                return []
            if isinstance(event, agents.AgentStream):
                return self._stream(event.delta, event.thinking_delta, event.current_agent_name)
            if isinstance(event, agents.AgentOutput):
                return [("AgentOutput", dumps(self._output(event)))]
            if isinstance(event, agents.ToolCallResult):
                return [("ToolCallResult", dumps(self._result(event)))]
            if isinstance(event, agents.ToolCall):
                if event.tool_id in self._tools:
                    return []
                self._tools.add(event.tool_id)
                return [("ToolCall", dumps(self._call(event.tool_id, event.tool_name, tool_kwargs(event))))]
            if isinstance(event, agents.AgentStreamStructuredOutput):
                output, truncated = bounded_json(event.output, self._max_field_bytes, _TRUNCATED)
                return [("AgentStreamStructuredOutput", dumps(_marked({"output": output}, truncated)))]
        ag = agui_events()
        if ag is not None and isinstance(event, ag.BaseEvent):
            return [(event_name(event), event.model_dump_json(by_alias=True))]
        return [(event_name(event), dumps(fields(event)))]

    def abort(self, error: BaseException) -> Iterable[Frame]:
        if isinstance(error, asyncio.CancelledError):
            return [("RunAborted", dumps({"reason": "cancelled"}))]
        payload: dict[str, Any] = {"reason": "error", "message": self._message(error)}
        if self._expose_errors:
            payload["type"] = type(error).__name__
        return [("RunAborted", dumps(payload))]

    def fit(self, frame: Frame, fits: Fits) -> Iterable[Frame]:
        name, data = frame
        try:
            value: Any = json.loads(data)
        except ValueError:
            return []
        if not isinstance(value, dict):
            return []
        if name == "AgentStream":
            return self._split_stream(value, fits)  # pyright: ignore[reportUnknownArgumentType]
        fitted = shrink_strings(value, lambda shrunk: (name, dumps(_marked(shrunk, True))), fits, _TRUNCATED, _KEEP)
        return [] if fitted is None else [fitted]

    def _stream(self, delta: str, thinking: str | None, agent: str) -> list[Frame]:
        if not delta and not thinking:
            return []  # a chunk that only carries tool-call deltas
        payload: dict[str, Any] = {"delta": delta, "agent": agent}
        if thinking:
            payload["thinking_delta"] = thinking
        return [("AgentStream", dumps(payload))]

    def _split_stream(self, value: dict[str, Any], fits: Fits) -> list[Frame]:
        """A long delta becomes several frames whose deltas a page concatenates as it would the original."""
        agent = value.get("agent", "")
        pieces: list[Frame] = []
        for key in ("thinking_delta", "delta"):
            text = value.get(key)
            if not isinstance(text, str) or not text:
                continue

            def build(piece: str, key: str = key) -> Frame:
                payload = {"delta": piece, "agent": agent} if key == "delta" else {"delta": "", "agent": agent}
                if key == "thinking_delta":
                    payload[key] = piece
                return ("AgentStream", dumps(payload))

            split = split_text(text, build, fits)
            if split is None:
                return []
            pieces.extend(split)
        return pieces

    def _text(self, text: str) -> tuple[str, bool]:
        short = cut(text, self._max_field_bytes)
        return (short + _TRUNCATED, True) if short != text else (text, False)

    def _call(self, tool_id: str, tool_name: str, kwargs: dict[str, Any]) -> dict[str, Any]:
        bounded, truncated = bounded_json(kwargs, self._max_field_bytes, _TRUNCATED)
        return _marked({"tool_id": tool_id, "tool_name": tool_name, "tool_kwargs": bounded}, truncated)

    def _output(self, event: AgentOutput) -> dict[str, Any]:
        content, truncated = self._text(event.response.content or "")
        calls = [
            {"tool_id": call.tool_id, "tool_name": call.tool_name, "tool_kwargs": call.tool_kwargs}
            for call in event.tool_calls
        ]
        bounded, calls_truncated = bounded_json(calls, self._max_field_bytes, _TRUNCATED)
        payload = {"agent": event.current_agent_name, "content": content, "tool_calls": bounded}
        return _marked(payload, truncated or calls_truncated)

    def _result(self, event: ToolCallResult) -> dict[str, Any]:
        output = event.tool_output
        is_error = bool(output.is_error)
        if is_error and not self._expose_errors:
            content, truncated = MASKED_ERROR, False
        else:
            content, truncated = self._text(str(output.content))
        payload = {
            "tool_id": event.tool_id,
            "tool_name": event.tool_name,
            "content": content,
            "is_error": is_error,
            "return_direct": bool(event.return_direct),
        }
        return _marked(payload, truncated)

    def _terminal(self, event: StopEvent) -> dict[str, Any]:
        if isinstance(event, WorkflowFailedEvent):
            kind, message = failure(event)
            payload: dict[str, Any] = {"step_name": event.step_name, "message": self._message(message)}
            if self._expose_errors:
                payload["type"] = kind
            return payload
        if isinstance(event, WorkflowTimedOutEvent):
            return {"timeout": event.timeout, "active_steps": list(event.active_steps)}
        if isinstance(event, WorkflowCancelledEvent):
            return {}
        if type(event) is StopEvent:
            return self._result_value(event.result)
        return fields(event)

    def _result_value(self, result: Any) -> dict[str, Any]:
        """The run's result: JSON values as they are, anything else (an ``AgentOutput``, a model) as its text."""
        if result is None or isinstance(result, (bool, int, float)):
            return {"result": result}
        if isinstance(result, (dict, list)):
            bounded, truncated = bounded_json(result, self._max_field_bytes, _TRUNCATED)
            return _marked({"result": bounded}, truncated)
        text, truncated = self._text(str(result))
        return _marked({"result": text}, truncated)

    def _message(self, error: object) -> str:
        text = str(error)
        return text if self._expose_errors and text else MASKED_ERROR


def _marked(payload: dict[str, Any], truncated: bool) -> dict[str, Any]:
    return {**payload, "truncated": True} if truncated else payload
