"""LangGraph v2 stream parts as frame data: JSON as LangGraph Server serializes it, with trimmed metadata."""

from __future__ import annotations

import dataclasses
import datetime
import decimal
import enum
import json
import uuid
from collections.abc import Callable, Mapping, Sequence
from pathlib import PurePath
from typing import Any, Literal, cast

from langchain_core.messages import ToolMessage

#: The modes ``relay`` requests and publishes unless told otherwise. ``values`` is left out: each part repeats the
#: whole state, which grows with the thread.
DEFAULT_STREAM_MODE: tuple[str, ...] = ("messages", "updates", "custom", "tools")

#: The metadata keys of a ``messages`` part that the browser clients read: ``useStream`` reads
#: ``langgraph_checkpoint_ns`` and ``lc_subagent_tool_call_id``, ``@ai-sdk/langchain`` ``langgraph_step``.
SLIM_METADATA: tuple[str, ...] = (
    "langgraph_node",
    "langgraph_step",
    "langgraph_checkpoint_ns",
    "lc_agent_name",
    "lc_subagent_tool_call_id",
    "ls_model_name",
)

#: What a browser reads in place of an error's message, unless errors are exposed.
MASKED_MESSAGE = "An error occurred."
TRUNCATED_MARKER = "…[truncated]"

Metadata = Literal["slim", "full"] | Sequence[str]


def _default(value: object) -> object:
    if isinstance(value, type):
        return repr(value)
    # Pydantic objects (messages among them) as model_dump(), as LangGraph Server does.
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump()
    if dataclasses.is_dataclass(value):
        # One level only: json recurses into the fields and calls this again where it must.
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    as_dict = getattr(value, "_asdict", None)
    if callable(as_dict):
        return as_dict()
    if isinstance(value, (set, frozenset)):
        return list(cast("set[object]", value))
    if isinstance(value, BaseException):
        return {"error": type(value).__name__}
    # The standard types as LangGraph Server's orjson writes them, and a Decimal as pydantic's JSON mode does.
    if isinstance(value, (datetime.datetime, datetime.date, datetime.time)):
        return value.isoformat()
    if isinstance(value, datetime.timedelta):
        return value.total_seconds()
    if isinstance(value, enum.Enum):
        return cast("object", value.value)
    if isinstance(value, (uuid.UUID, decimal.Decimal, PurePath)):
        return str(value)
    # One odd value in a state never costs the frame.
    return repr(value)


_SEPARATORS = (",", ":")


def _null(constant: str) -> None:
    return None


def dumps(value: object) -> str:
    """Compact JSON, non-ASCII unescaped, with LangGraph Server's handling of pydantic objects and dataclasses. A float
    JSON cannot hold (NaN, an infinity) is ``null``, as orjson writes it: a browser's ``JSON.parse`` refuses the bare
    constants, and with them the whole stream."""
    try:
        return json.dumps(value, default=_default, ensure_ascii=False, separators=_SEPARATORS, allow_nan=False)
    except ValueError:
        # Rare: the bare constants are written, then read back as null. A circular value raises here again.
        text = json.dumps(value, default=_default, ensure_ascii=False, separators=_SEPARATORS)
        return json.dumps(json.loads(text, parse_constant=_null), ensure_ascii=False, separators=_SEPARATORS)


def check_metadata(metadata: Metadata) -> None:
    if isinstance(metadata, str):
        if metadata not in ("slim", "full"):
            raise ValueError('metadata is "slim", "full" or a sequence of keys.')
    elif not all(isinstance(key, str) for key in metadata):  # pyright: ignore[reportUnnecessaryIsInstance]
        raise ValueError("metadata keys are strings.")


def _trim(meta: Mapping[str, Any], metadata: Metadata) -> Mapping[str, Any]:
    if metadata == "full":
        return meta
    keys = SLIM_METADATA if metadata == "slim" else metadata
    return {key: meta[key] for key in keys if key in meta}


def mask_failed_tools(value: object, depth: int = 4) -> object:
    """``value`` with the content of each failed tool's ``ToolMessage`` (``status="error"``, whose content is the
    error's text) replaced, in the mappings and lists of a state or an update. What holds none is returned as it is."""
    if isinstance(value, ToolMessage):
        return value.model_copy(update={"content": MASKED_MESSAGE}) if value.status == "error" else value
    if depth == 0:
        return value
    if isinstance(value, Mapping):
        fields = cast("Mapping[object, object]", value)
        masked = {key: mask_failed_tools(item, depth - 1) for key, item in fields.items()}
        return masked if any(masked[key] is not item for key, item in fields.items()) else fields
    if isinstance(value, (list, tuple)):
        items = cast("Sequence[object]", value)
        cut = [mask_failed_tools(item, depth - 1) for item in items]
        return cut if any(new is not old for new, old in zip(cut, items, strict=True)) else items
    return value


def _masked_data(kind: str, data: Any) -> Any:
    """The part's data with the error text a browser must not read replaced."""
    if kind == "custom":  # the application's own data
        return cast("object", data)
    data = mask_failed_tools(data)
    if not isinstance(data, Mapping):
        return cast("object", data)
    fields = cast("Mapping[str, Any]", data)
    if kind == "tools" and fields.get("event") == "tool-error" and "message" in fields:
        return {**fields, "message": MASKED_MESSAGE}
    if kind == "tasks" and fields.get("error") is not None:
        return {**fields, "error": MASKED_MESSAGE}
    return fields


def shape(part: Mapping[str, Any], *, metadata: Metadata = "slim", expose_errors: bool = False) -> dict[str, Any]:
    """The JSON-ready form of a part: ``{"type", "ns", "data"}`` (and ``interrupts`` when a ``values`` part has
    them), metadata trimmed, error text masked."""
    kind = str(part["type"])
    data = part["data"]
    if kind == "messages" and isinstance(data, (tuple, list)) and len(cast("Sequence[Any]", data)) == 2:
        message, meta = cast("Sequence[Any]", data)
        if not expose_errors:
            message = mask_failed_tools(message, 0)
        data = [message, _trim(cast("Mapping[str, Any]", meta), metadata) if isinstance(meta, Mapping) else meta]
    elif not expose_errors:
        data = _masked_data(kind, data)
    shaped: dict[str, Any] = {"type": kind, "ns": list(part.get("ns", ())), "data": data}
    if part.get("interrupts"):
        shaped["interrupts"] = part["interrupts"]
    return shaped


def encode_part(part: Mapping[str, Any], *, metadata: Metadata = "slim", expose_errors: bool = False) -> str:
    """The frame data for one LangGraph v2 stream part, as ``relay`` publishes it.

    ``metadata`` keeps the keys of a ``messages`` part's metadata that browser clients read (``"slim"``), all of
    them (``"full"``) or the ones listed. Unless ``expose_errors``, the message of a ``tool-error`` part, the error
    of a ``tasks`` result and the content of a failed tool's ``ToolMessage`` are replaced by a fixed text.
    """
    check_metadata(metadata)
    return dumps(shape(part, metadata=metadata, expose_errors=expose_errors))


def error_part(error: BaseException, *, expose: bool = False) -> str:
    """The ``error`` part that ends a failed run: the exception's type name, and its message only if ``expose``."""
    message = str(error) if expose else MASKED_MESSAGE
    return dumps({"type": "error", "ns": [], "data": {"error": type(error).__name__, "message": message}})


#: The ``error`` part of a cancelled run. The name tells a browser that it was not a failure of the run.
CANCELLED_PART = dumps({"type": "error", "ns": [], "data": {"error": "Cancelled", "message": "run cancelled"}})


def metadata_part(config: Mapping[str, Any] | None) -> str:
    """The first part of a stream, as LangGraph Server sends it: the run and thread a browser reloads after
    ``iq:gone``."""
    config = config or {}
    configurable = config.get("configurable")
    thread_id = cast("Mapping[str, Any]", configurable).get("thread_id") if isinstance(configurable, Mapping) else None
    run_id = config.get("run_id")
    data = {
        "run_id": None if run_id is None else str(run_id),
        "thread_id": None if thread_id is None else str(thread_id),
    }
    return dumps({"type": "metadata", "ns": [], "data": data})


def truncated_messages_part(shaped: Mapping[str, Any], fits: Callable[[str], bool]) -> str | None:
    """A ``messages`` part whose message content is cut so that its frame fits, ending in a marker and with
    ``"truncated": true`` in its metadata; ``None`` when its content is not a string or nothing of it fits."""
    if shaped.get("type") != "messages":
        return None
    message, meta = cast("Sequence[object]", shaped["data"])
    plain = cast("object", message) if isinstance(message, Mapping) else _default(message)
    if not isinstance(plain, Mapping):
        return None
    fields = cast("Mapping[str, Any]", plain)
    content = fields.get("content")
    if not isinstance(content, str):
        return None
    meta = {**cast("Mapping[str, Any]", meta), "truncated": True} if isinstance(meta, Mapping) else {"truncated": True}

    def frame(keep: int) -> str:
        cut = {**fields, "content": content[:keep] + TRUNCATED_MARKER}
        return dumps({**shaped, "data": [cut, meta]})

    if not fits(frame(0)):
        return None
    return frame(longest_fit(len(content), lambda keep: fits(frame(keep))))


#: Keys whose string values a structural cut keeps whole: they name or identify, and a cut one would not match.
_IDENTIFIERS = frozenset(
    {
        "id",
        "type",
        "name",
        "reason",
        "action_name",
        "tool_call_id",
        "toolCallId",
        "threadId",
        "runId",
        "messageId",
        "subagentRunId",
    }
)


def _cut(text: str, keep: int) -> str:
    # A string no longer than its cut stays whole: the size of a cut grows with keep, which the search relies on.
    return text if len(text) <= keep + len(TRUNCATED_MARKER) else text[:keep] + TRUNCATED_MARKER


def cut_strings(value: object, keep: int, key: object = None) -> object:
    """A JSON value with every long string cut to ``keep`` characters and a marker, except those of identifying
    keys; the structure, the keys and every other value stay."""
    if isinstance(value, str):
        return value if key in _IDENTIFIERS else _cut(value, keep)
    if isinstance(value, Mapping):
        return {name: cut_strings(item, keep, name) for name, item in cast("Mapping[str, object]", value).items()}
    if isinstance(value, list):
        return [cut_strings(item, keep, key) for item in cast("list[object]", value)]
    return value


def _longest_string(value: object) -> int:
    if isinstance(value, str):
        return len(value)
    if isinstance(value, Mapping):
        return max(map(_longest_string, cast("Mapping[str, object]", value).values()), default=0)
    if isinstance(value, list):
        return max(map(_longest_string, cast("list[object]", value)), default=0)
    return 0


def cut_to_fit(value: object, fits: Callable[[object], bool]) -> object | None:
    """``value`` (plain JSON) with its long strings cut as little as ``fits`` allows, or ``None`` when it does not fit
    even with every one of them cut."""
    if not fits(cut_strings(value, 0)):
        return None
    return cut_strings(value, longest_fit(_longest_string(value), lambda keep: fits(cut_strings(value, keep))))


def truncated_interrupt_part(shaped: Mapping[str, Any], fits: Callable[[str], bool]) -> str | None:
    """An ``updates`` part that pauses the run, reduced to its interrupts: each keeps its ``id`` and its ``value`` with
    the long strings cut to fit (names and ids whole, so that a client still finds the tool call a human must
    approve), and has ``"truncated": true``; a value that does not fit even so is ``null``. The part is the only one
    that tells the browser the run waits for a human, so it is never dropped while its ids fit. ``None`` when the
    part holds no interrupt or even its ids do not fit."""
    data = shaped.get("data")
    if shaped.get("type") != "updates" or not isinstance(data, Mapping):
        return None
    pending = cast("Mapping[str, Any]", data).get("__interrupt__")
    if not isinstance(pending, (list, tuple)) or not pending:
        return None
    # Objects as the frame writes them, so that the cut works on JSON only.
    plain = json.loads(dumps(cast("Sequence[object]", pending)))
    interrupts: list[dict[str, Any]] = []
    for item in cast("list[object]", plain):
        fields = cast("dict[str, Any]", item) if isinstance(item, dict) else {}
        interrupts.append({"value": fields.get("value"), "id": fields.get("id"), "truncated": True})

    def frame(reduced: object) -> str:
        return dumps({**shaped, "data": {"__interrupt__": reduced}})

    cut = cut_to_fit(interrupts, lambda reduced: fits(frame(reduced)))
    if cut is not None:
        return frame(cut)
    bare = [{**fields, "value": None} for fields in interrupts]
    return frame(bare) if fits(frame(bare)) else None


def longest_fit(length: int, fits: Callable[[int], bool]) -> int:
    """The largest ``keep`` in ``0..length`` for which ``fits(keep)``, given that ``fits(0)``. A search, since escapes
    and multi-byte characters make the size of a cut hard to predict."""
    low, high = 0, length
    while low < high:
        middle = (low + high + 1) // 2
        if fits(middle):
            low = middle
        else:
            high = middle - 1
    return low
