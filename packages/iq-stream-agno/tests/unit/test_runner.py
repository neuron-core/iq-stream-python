"""start_run() and continue_run(): a run in a task of its own, its ticket at once, its summary at the end."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from iq_stream import IQClient
from recorded import RecordingClient, names, types
from scripted_model import SLOW, weather_agent

from iq_stream_agno import RunSummary, StreamHandle, continue_run, running, start_run


async def test_start_run_answers_before_the_first_event_and_the_task_publishes_the_run() -> None:
    client = RecordingClient()
    agent = weather_agent(db=InMemoryDb())
    handle = start_run(client, agent, "Weather?", session_id="s-start")
    (publisher,) = client.recorders
    assert publisher.frames == [] and handle.task in running()
    assert handle.stream_id == publisher.stream_id and handle.session_id == "s-start"
    assert handle.url == f"https://stream.example.com/{handle.stream_id}?token={handle.token}"
    assert set(handle.to_dict()) == {"streamId", "token", "url", "expiresAt", "runId", "sessionId"}
    assert client.verify("sub", handle.token, handle.stream_id)

    summary = await asyncio.wait_for(handle.task, 10)
    assert isinstance(summary, RunSummary)
    assert (summary.outcome, summary.status, summary.run_id) == ("end", "completed", handle.run_id)
    assert summary.frames == len(publisher.frames) and summary.result is not None and summary.result.closed
    assert names(publisher)[0] == "RunStarted" and publisher.outcome == "end"
    stored: Any = await agent.aget_run_output(handle.run_id, session_id="s-start")
    assert stored.metadata["iq_stream_id"] == handle.stream_id
    assert handle.task not in running()


async def test_start_run_keeps_the_caller_s_metadata_and_options() -> None:
    client = RecordingClient()
    agent = weather_agent(db=InMemoryDb())
    handle = start_run(
        client,
        agent,
        "Weather?",
        run_id="chosen-run",
        protocol="agui",
        metadata={"tenant": "acme"},
        publish_options={"exclude": ("ModelRequestStarted",)},
    )
    summary = await asyncio.wait_for(handle.task, 10)
    assert handle.run_id == summary.run_id == "chosen-run"
    (publisher,) = client.recorders
    assert types(publisher)[0] == "RUN_STARTED" and types(publisher)[-1] == "RUN_FINISHED"
    stored: Any = await agent.aget_run_output("chosen-run", session_id=handle.session_id)
    assert stored.metadata == {"tenant": "acme", "iq_stream_id": handle.stream_id}


async def test_a_run_that_cannot_start_fails_its_stream_and_says_why() -> None:
    client = RecordingClient()

    class Broken(Agent):
        def arun(self, *args: Any, **kwargs: Any) -> Any:  # type: ignore[override]
            raise TypeError("bad argument")

    handle = start_run(client, Broken(model=None, telemetry=False), "Weather?")
    summary = await asyncio.wait_for(handle.task, 10)
    assert summary.status == "error" and isinstance(summary.error, TypeError)
    assert client.recorders[0].outcome == "fail"


async def test_cancelling_the_task_cancels_the_run_and_fails_the_stream() -> None:
    SLOW["seconds"] = 1.0
    client = RecordingClient()
    handle = start_run(client, weather_agent(tool_name="slow_weather"), "Weather?")
    (publisher,) = client.recorders
    for _ in range(500):
        if "ToolCallStarted" in names(publisher):
            break
        await asyncio.sleep(0.01)
    handle.task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await handle.task
    assert publisher.outcome == "fail"


async def test_continue_run_publishes_the_next_leg_on_a_new_stream() -> None:
    client = RecordingClient()
    agent = weather_agent(tool_name="book_trip", answer="Your trip is booked.", db=InMemoryDb())
    first = start_run(client, agent, "Book Paris", session_id="s-continue")
    summary = await asyncio.wait_for(first.task, 10)
    assert summary.status == "paused" and summary.outcome == "end"
    paused: Any = await agent.aget_run_output(first.run_id, session_id="s-continue")
    for requirement in paused.requirements:
        requirement.confirm()
    second = continue_run(client, agent, run_id=first.run_id, session_id="s-continue", requirements=paused.requirements)
    assert second.stream_id != first.stream_id and second.run_id == first.run_id
    summary = await asyncio.wait_for(second.task, 10)
    assert summary.status == "completed" and names(client.recorders[1])[0] == "RunContinued"


async def test_start_run_needs_a_running_loop_and_a_public_url() -> None:
    def outside_a_loop() -> StreamHandle:
        return start_run(RecordingClient(), weather_agent(), "Weather?")

    with pytest.raises(RuntimeError):
        await asyncio.to_thread(outside_a_loop)
    socket_client = IQClient("http+unix:///tmp/iq.sock", "demo-app", "test-app-secret-0123456789abcdef0123")
    with pytest.raises(Exception, match="public_url"):
        start_run(socket_client, weather_agent(), "Weather?")
