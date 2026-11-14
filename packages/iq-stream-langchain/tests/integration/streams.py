"""What the integration tests share: a subscriber started before the run, and checks of what it received."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from ag_ui.core import RunAgentInput, UserMessage
from ag_ui_langgraph import LangGraphAgent
from fakes import USER
from iq_stream import IQClient, Ticket
from iq_stream.testing import BackgroundSubscription, Subscription, subscribe_in_background

from iq_stream_langchain.agui import arelay_agui


def watch(client: IQClient, url: str) -> tuple[Ticket, BackgroundSubscription]:
    """A ticket, and a subscriber that has opened the stream before its publisher exists."""
    ticket = client.ticket()
    subscription = subscribe_in_background(url, ticket.stream_id, ticket.token, timeout=30)
    return ticket, subscription


def parts(received: Subscription) -> list[dict[str, Any]]:
    """The frames as the parts they carry, after the checks every stream must pass: unnamed frames, ids 1..n
    without a gap or a repeat, and a terminal event that counts them all."""
    assert all(frame.event == "message" for frame in received.frames)
    assert [frame.id for frame in received.frames] == [str(n) for n in range(1, len(received.frames) + 1)]
    assert received.terminal is not None, "the stream ended without a terminal event"
    assert received.terminal.data.get("lastEventId") == len(received.frames)
    return [json.loads(frame.data) for frame in received.frames]


def recording(received: Subscription) -> dict[str, Any]:
    """What the subscriber received, for the front-end checks to replay."""
    terminal = received.terminal
    return {
        "frames": [{"id": frame.id, "event": frame.event, "data": frame.data} for frame in received.frames],
        "terminal": None if terminal is None else {"event": terminal.event, "data": terminal.data},
    }


def run_input() -> RunAgentInput:
    return RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        tools=[],
        context=[],
        forwarded_props={},
        messages=[UserMessage(id="u1", role="user", content=USER["messages"][0]["content"])],
    )


async def publish_agui(client: IQClient, url: str, graph: Any, *, cancel_after: float | None = None) -> Subscription:
    """Runs the graph through ag-ui-langgraph and the relay, and returns what a subscriber received."""
    ticket, subscription = watch(client, url)
    agent = LangGraphAgent(name="weather", graph=graph, emit_raw_events=False)

    async def run() -> None:
        async with client.apublisher(ticket.stream_id) as publisher:
            async for _ in arelay_agui(publisher, agent, run_input()):
                pass

    task = asyncio.create_task(run())
    if cancel_after is not None:
        await asyncio.sleep(cancel_after)
        task.cancel()
    with pytest.raises(asyncio.CancelledError) if cancel_after is not None else contextlib.nullcontext():
        await task
    return await subscription
