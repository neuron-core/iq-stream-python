"""Agno's own event JSON as frames: ``event.to_dict()``, as AgentOS writes it after ``data:``, and the shrink policy for
a payload over the line limit."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from agno.run.base import BaseRunOutputEvent
from agno.utils.serialize import json_serializer
from iq_stream import FrameTooLarge

from ._common import EVENT_NAME, MASKED_ERROR, Fits, Frame, Report, event_name, longest_prefix

Transform = Callable[[BaseRunOutputEvent], "Mapping[str, Any] | None"]
"""Turns an event into the dictionary to publish, or ``None`` to publish nothing for it."""

#: Error events and the member that carries the error's text.
_ERROR_TEXT = {
    "RunError": "content",
    "TeamRunError": "content",
    "WorkflowError": "error",
    "StepError": "error",
    "ToolCallError": "error",
    "TeamToolCallError": "error",
}
_TOOL_EVENTS = frozenset({"ToolCallCompleted", "TeamToolCallCompleted", "ToolCallError", "TeamToolCallError"})

#: What the shrink policy removes, group by group, until the line fits: nested runs, media, then context.
_REMOVABLE = (
    ("member_responses", "step_executor_runs", "step_results", "step_response"),
    ("images", "videos", "audio", "files", "response_audio", "image"),
    ("additional_input", "reasoning_messages", "references", "session_state"),
)
TRUNCATED = "iq_truncated"


def dumps(payload: Mapping[str, Any]) -> str:
    """The JSON that AgentOS's ``format_sse_event`` writes: compact, non-ASCII unescaped."""
    return json.dumps(payload, separators=(",", ":"), default=json_serializer, ensure_ascii=False)


def masked(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The payload with the error text of an error event, or of a failed tool's result, replaced."""
    field = _ERROR_TEXT.get(name)
    if field is not None:
        if payload.get(field) is not None:
            payload[field] = MASKED_ERROR
        payload.pop("additional_data", None)
    tool: object = payload.get("tool")
    if name in _TOOL_EVENTS and isinstance(tool, dict):
        tool = cast("dict[str, Any]", tool)
        if tool.get("tool_call_error") and tool.get("result") is not None:
            payload["tool"] = {**tool, "result": MASKED_ERROR}
    return payload


class NativeEncoder:
    """One run's events as Agno's own JSON, one frame per event."""

    def __init__(
        self,
        *,
        event_names: bool,
        transform: Transform | None,
        expose_errors: bool,
        shrink: bool,
        fits: Fits,
        size: Callable[[str, str | None], int],
        limit: int,
        report: Report,
    ) -> None:
        self._event_names = event_names
        self._transform = transform
        self._expose_errors = expose_errors
        self._shrink = shrink
        self._fits = fits
        self._size = size
        self._limit = limit
        self._report = report

    def encode(self, event: BaseRunOutputEvent) -> list[Frame]:
        name = event_name(event)
        if self._transform is None:
            payload: dict[str, Any] = event.to_dict()
        else:
            mapping = self._transform(event)
            if mapping is None:
                return []
            payload = dict(mapping)
        if not self._expose_errors:
            payload = masked(name, payload)
        frame_event = name if self._event_names and EVENT_NAME.fullmatch(name) else None
        data = dumps(payload)
        if self._fits(data, frame_event):
            return [(frame_event, data)]
        size = self._size(data, frame_event)
        shrunk = shrink(payload, lambda text: self._fits(text, frame_event)) if self._shrink else None
        if shrunk is None:
            self._report(FrameTooLarge(size, self._limit))
            return []
        return [(frame_event, shrunk)]


def shrink(payload: dict[str, Any], fits: Callable[[str], bool]) -> str | None:
    """Removes members in a fixed order, then cuts tool results and text, until the JSON fits; lists what it removed
    or cut under ``iq_truncated``. ``None`` when even that is not enough."""
    payload = dict(payload)
    truncated: list[str] = []

    def encoded() -> str:
        return dumps({**payload, TRUNCATED: truncated})

    for group in _REMOVABLE:
        for key in group:
            if key in payload:
                del payload[key]
                truncated.append(key)
        if truncated and fits(encoded()):
            return encoded()

    # The text that remains, longest first within each kind: tool results, then the event's own text.
    tool: object = payload.get("tool")
    if isinstance(tool, dict):
        payload["tool"] = dict(cast("dict[str, Any]", tool))
    tools: object = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"] = [
            dict(cast("dict[str, Any]", item)) if isinstance(item, dict) else item for item in cast("list[Any]", tools)
        ]
    for holder, key, label in _texts(payload):
        if fits(encoded()):
            return encoded()
        value: object = holder.get(key)
        if value is None:
            continue
        if label not in truncated:
            truncated.append(label)
        if not isinstance(value, str):
            del holder[key]
            continue

        def fits_cut(prefix: str, holder: dict[str, Any] = holder, key: str = key) -> bool:
            holder[key] = prefix
            return fits(encoded())

        length = longest_prefix(value, fits_cut)
        holder[key] = value[: max(length, 0)]
    return encoded() if fits(encoded()) else None


def _texts(payload: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    found: list[tuple[dict[str, Any], str, str]] = []
    tool: object = payload.get("tool")
    if isinstance(tool, dict):
        found.append((cast("dict[str, Any]", tool), "result", "tool.result"))
    tools: object = payload.get("tools")
    if isinstance(tools, list):
        items = [cast("dict[str, Any]", item) for item in cast("list[Any]", tools) if isinstance(item, dict)]
        items.sort(key=lambda item: len(str(item.get("result") or "")), reverse=True)
        found.extend((item, "result", "tools[].result") for item in items)
    found.extend((payload, key, key) for key in ("content", "reasoning_content"))
    return found
