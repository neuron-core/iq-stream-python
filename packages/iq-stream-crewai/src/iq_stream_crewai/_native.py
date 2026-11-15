"""``protocol="crewai"``: one frame per CrewAI event, named after its type, with a compact JSON of its members."""

from __future__ import annotations

import re

from crewai.events.base_events import BaseEvent

from ._limits import Fitter, Frame
from ._snapshot import Snapshot, Snapshotter

_EVENT_NAME = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_DELTAS = frozenset({"llm_stream_chunk", "llm_thinking_chunk"})
# What may be cut when a frame is over the limit, in no particular order: the longest goes first.
_CUTTABLE = (
    *("output", "error", "message", "feedback", "tool_args", "task_name", "usage"),
    *("agent_role", "crew_name", "flow_name", "method_name", "tool_name", "model"),
)


class NativeEncoder:
    def __init__(self, snapshotter: Snapshotter, fitter: Fitter) -> None:
        self._snapshot = snapshotter
        self._fit = fitter

    def snapshot(self, source: object, event: BaseEvent) -> Snapshot | None:
        return self._snapshot(source, event)

    def start(self) -> list[Frame]:
        return []

    def frames(self, snapshot: Snapshot) -> list[Frame]:
        if not self._snapshot.admit(snapshot):
            return []
        name = snapshot.type if _EVENT_NAME.fullmatch(snapshot.type) else None
        payload = {"seq": snapshot.seq, **snapshot.fields} if snapshot.seq is not None else dict(snapshot.fields)
        if snapshot.type in _DELTAS:
            return self._fit.split(payload, "chunk", name)
        return self._fit.cut(payload, _CUTTABLE, name)

    def finish(self, error: BaseException | None) -> list[Frame]:
        return []
