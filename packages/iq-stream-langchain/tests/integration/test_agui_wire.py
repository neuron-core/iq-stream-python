"""The AG-UI path through the real server: every frame a valid AG-UI event, the run's end in the close line."""

from __future__ import annotations

import json
from typing import Any

from ag_ui.core import Event
from fakes import failing_agent, weather_agent
from iq_stream import IQClient
from iq_stream.testing import Subscription, TestServers
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter
from streams import parts, publish_agui

from iq_stream_langchain import MASKED_MESSAGE

EVENT = TypeAdapter[Any](Event)


def events(received: Subscription) -> list[dict[str, Any]]:
    published = parts(received)
    for frame in received.frames:
        EVENT.validate_json(frame.data)
    return published


async def test_an_ag_ui_run_arrives_as_valid_events_and_ends(iq_client: IQClient, iq_servers: TestServers) -> None:
    received = await publish_agui(iq_client, iq_servers.default.url, weather_agent(checkpointer=InMemorySaver()))
    published = events(received)
    assert received.terminal is not None and received.terminal.event == "iq:end"
    types = [event["type"] for event in published]
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    for needed in ("TOOL_CALL_START", "TOOL_CALL_END", "TOOL_CALL_RESULT", "MESSAGES_SNAPSHOT"):
        assert needed in types


async def test_a_failing_ag_ui_run_ends_with_run_error_then_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    received = await publish_agui(iq_client, iq_servers.default.url, failing_agent(checkpointer=InMemorySaver()))
    published = events(received)
    assert published[-1] == {"type": "RUN_ERROR", "message": MASKED_MESSAGE}
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert received.terminal.data["reason"] == "publisher"


async def test_a_cancelled_ag_ui_run_finishes_cancelled_then_iq_fail(
    iq_client: IQClient, iq_servers: TestServers
) -> None:
    graph = weather_agent(token_delay=0.05, checkpointer=InMemorySaver())
    received = await publish_agui(iq_client, iq_servers.default.url, graph, cancel_after=0.3)
    published = events(received)
    assert published[-1]["type"] == "RUN_FINISHED" and published[-1]["outcome"] == {"type": "cancelled"}
    assert received.terminal is not None and received.terminal.event == "iq:fail"
    assert json.loads(received.frames[0].data)["type"] == "RUN_STARTED"
