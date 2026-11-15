"""What the integration tests share: clients of the two servers and checks of what a real subscriber received."""

from __future__ import annotations

import json
from typing import Any

from iq_stream import IQClient
from iq_stream.testing import Subscription, TestServers


def short_client(servers: TestServers, **options: Any) -> IQClient:
    """A client of the short server (2 s read timeout, 1,024-byte frames, 8,192-byte streams)."""
    settings: dict[str, Any] = {"min_open": 0.0, "max_frame_bytes": 1024, **options}
    return IQClient(servers.short.url, servers.app_key, servers.app_secret, **settings)


def rows(subscription: Subscription) -> list[dict[str, Any]]:
    return [json.loads(frame.data) for frame in subscription.frames]


def delivered(subscription: Subscription, terminal: str) -> list[dict[str, Any]]:
    """The frames, after checking that they arrived once each, in order, before the expected terminal event."""
    ids = [int(frame.id) for frame in subscription.frames]
    assert ids == list(range(1, len(ids) + 1)), f"ids are not 1..n: {ids}"
    assert subscription.terminal is not None, "the stream did not end"
    assert subscription.terminal.event == terminal, subscription.terminal
    if terminal != "iq:gone":
        assert subscription.terminal.data.get("lastEventId") == len(ids)
    return rows(subscription)


def recording(subscription: Subscription, protocol: str) -> dict[str, Any]:
    """What the front-end check replays: the frames and the terminal event exactly as the subscriber got them."""
    assert subscription.terminal is not None
    return {
        "protocol": protocol,
        "frames": [{"id": frame.id, "event": frame.event, "data": frame.data} for frame in subscription.frames],
        "terminal": {"event": subscription.terminal.event, "data": subscription.terminal.data},
    }
