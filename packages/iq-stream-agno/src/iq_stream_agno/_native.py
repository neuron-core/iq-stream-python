"""Agno's own event JSON as frames: ``event.to_dict()``, as AgentOS writes it after ``data:``, and the shrink policy for
a payload over the line limit."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from typing import Any, cast

from agno.run.base import BaseRunOutputEvent
from agno.utils.serialize import json_serializer
from iq_stream import FrameTooLarge

from ._common import ERROR_TEXT, EVENT_NAME, MASKED_ERROR, Fits, Frame, Report, event_name, longest_prefix, mask_errors

Transform = Callable[[BaseRunOutputEvent], "Mapping[str, Any] | None"]
"""Turns an event into the dictionary to publish, or ``None`` to publish nothing for it."""

#: What the shrink policy removes, group by group, until the line fits: nested runs, the requirements of a paused
#: run (which repeat its ``tools``), media, then context.
_REMOVABLE = (
    ("member_responses", "step_executor_runs", "step_results", "step_response"),
    ("requirements",),
    ("images", "videos", "audio", "files", "response_audio", "image"),
    ("additional_input", "reasoning_messages", "references", "session_state"),
)
TRUNCATED = "iq_truncated"


def dumps(payload: Mapping[str, Any]) -> str:
    """The JSON that AgentOS's ``format_sse_event`` writes: compact, non-ASCII unescaped."""
    return json.dumps(payload, separators=(",", ":"), default=json_serializer, ensure_ascii=False)


def masked(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """The payload without error text: its own, found by the event's name (a transform may leave ``event`` out),
    and that of the runs nested in it (``mask_errors``)."""
    payload = cast("dict[str, Any]", mask_errors(payload))
    field = ERROR_TEXT.get(name)
    if field is not None:
        if payload.get(field) is not None:
            payload[field] = MASKED_ERROR
        payload.pop("additional_data", None)
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
    """Removes members in a fixed order, then cuts tool results and text and removes tool arguments, until the JSON
    fits; lists what it removed or cut under ``iq_truncated``. ``None`` when even that is not enough."""
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

    tool: object = payload.get("tool")
    if isinstance(tool, dict):
        payload["tool"] = dict(cast("dict[str, Any]", tool))
    tools: object = payload.get("tools")
    if isinstance(tools, list):
        payload["tools"] = [
            dict(cast("dict[str, Any]", item)) if isinstance(item, dict) else item for item in cast("list[Any]", tools)
        ]
    # Text is cut to what fits; an object is removed whole, and the text cut before it then grows back into the room.
    cut: list[tuple[dict[str, Any], str, str, str]] = []

    def grow(holder: dict[str, Any], key: str, text: str) -> None:
        def fits_cut(prefix: str) -> bool:
            holder[key] = prefix
            return fits(encoded())

        holder[key] = text[: max(longest_prefix(text, fits_cut), 0)]

    for holder, key, label in _places(payload):
        if fits(encoded()):
            break
        value: object = holder.get(key)
        if value is None:
            continue
        if label not in truncated:
            truncated.append(label)  # before the cut, which measures the line with it
        if isinstance(value, str):
            cut.append((holder, key, label, value))
            grow(holder, key, value)
        else:
            del holder[key]
            for earlier in reversed(cut):  # the text cut last grows first: a tool's result, cut first, takes the rest
                grow(earlier[0], earlier[1], earlier[3])
    partial = {label for holder, key, label, value in cut if holder[key] != value}
    whole = {label for _, _, label, _ in cut} - partial
    truncated[:] = [label for label in truncated if label not in whole]
    return encoded() if fits(encoded()) else None


def _places(payload: dict[str, Any]) -> list[tuple[dict[str, Any], str, str]]:
    """Where to cut, in order. A tool event's texts (the tool's result, and Agno's one-line summary, which repeats the
    tool's name and arguments) are cut longest first, so that a short one is not emptied while a long one keeps
    room, and its arguments go last; the text of an event that lists its tools, such as ``RunCompleted``, is the
    answer, cut last."""
    texts = [(payload, key, key) for key in ("content", "reasoning_content")]
    tool: object = payload.get("tool")
    if isinstance(tool, dict):
        tool = cast("dict[str, Any]", tool)
        own = [(tool, "result", "tool.result"), *texts]
        own.sort(key=lambda place: len(str(place[0].get(place[1]) or "")), reverse=True)
        return [*own, (tool, "tool_args", "tool.tool_args")]
    tools: object = payload.get("tools")
    listed = cast("list[Any]", tools) if isinstance(tools, list) else []
    items = [cast("dict[str, Any]", item) for item in listed if isinstance(item, dict)]
    items.sort(key=lambda item: len(str(item.get("result") or "")), reverse=True)
    found = [(item, "result", "tools[].result") for item in items]
    found.extend((item, "tool_args", "tools[].tool_args") for item in items)
    return [*found, *texts]


#: What an error event of each kind copies from the root event it stands in for.
_IDENTITY = ("agent_id", "agent_name", "team_id", "team_name", "workflow_id", "workflow_name")


def error_event(ending: BaseRunOutputEvent, text: str) -> BaseRunOutputEvent:
    """The error event of the root run's kind (``RunError``, ``TeamRunError``, ``WorkflowError``), with ``text``: what
    the native protocol publishes instead of a root event that turned out not to end the run, because the iterator
    raised after it or ``before_close`` failed."""
    name = event_name(ending)
    fields: dict[str, Any] = {key: getattr(ending, key, None) for key in ("run_id", "session_id")}
    if name.startswith("Workflow"):
        from agno.run.workflow import WorkflowErrorEvent as Error

        fields["error"] = text
    elif name.startswith("Team"):
        from agno.run.team import RunErrorEvent as Error

        fields["content"] = text
    else:
        from agno.run.agent import RunErrorEvent as Error

        fields["content"] = text
    known = Error.__dataclass_fields__
    fields.update({key: getattr(ending, key) for key in _IDENTITY if key in known and hasattr(ending, key)})
    return Error(**fields)
