"""What the integration tests assert on every stream a real subscriber received."""

from __future__ import annotations

import json
from typing import Any

from iq_stream.testing import Subscription


def assert_delivered(received: Subscription, terminal: str) -> None:
    """Frames numbered 1..n in order, none twice, then exactly the expected terminal event, whose lastEventId is n."""
    assert received.status == 200
    assert [frame.id for frame in received.frames] == [str(n) for n in range(1, len(received.frames) + 1)]
    assert received.terminal is not None and received.terminal.event == terminal
    assert received.terminal.data.get("lastEventId") == len(received.frames)


def named(received: Subscription) -> list[str]:
    return [frame.event for frame in received.frames]


def payloads(received: Subscription) -> list[dict[str, Any]]:
    return [json.loads(frame.data) for frame in received.frames]


def recording(received: Subscription, protocol: str) -> dict[str, Any]:
    """A subscription as tests/frontend/check.mjs replays it."""
    assert received.terminal is not None
    return {
        "protocol": protocol,
        "frames": [{"id": frame.id, "event": frame.event, "data": frame.data} for frame in received.frames],
        "terminal": {"event": received.terminal.event, "data": received.terminal.data},
    }
