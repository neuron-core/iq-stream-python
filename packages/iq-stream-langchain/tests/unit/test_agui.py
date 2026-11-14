"""The AG-UI path: the events of ag-ui-langgraph, one frame each, valid AG-UI on the wire whatever way the run ends."""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

import pytest
from ag_ui.core import Event, RunAgentInput, UserMessage
from ag_ui_langgraph import LangGraphAgent
from fakes import ASYNC_TOKENS, FINAL, FIRST, USER, FakeToolCallingModel, failing_agent, tool_call, weather_agent
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import TypeAdapter

from iq_stream_langchain import MASKED_MESSAGE, Oversize
from iq_stream_langchain.agui import arelay_agui

EVENT = TypeAdapter[Any](Event)


def run_input(thread_id: str = "thread-1") -> RunAgentInput:
    question = USER["messages"][0]["content"]
    return RunAgentInput(
        thread_id=thread_id,
        run_id="run-1",
        state={},
        tools=[],
        context=[],
        forwarded_props={},
        messages=[UserMessage(id="u1", role="user", content=question)],
    )


def agent_of(graph: Any) -> LangGraphAgent:
    return LangGraphAgent(name="weather", graph=graph, emit_raw_events=False)


def assert_balanced(published: list[dict[str, Any]]) -> None:
    """Every message, tool call and step that was opened is closed, as AG-UI clients require of a finished run."""
    for opener, closer, key in (
        ("TEXT_MESSAGE_START", "TEXT_MESSAGE_END", "messageId"),
        ("TOOL_CALL_START", "TOOL_CALL_END", "toolCallId"),
        ("STEP_STARTED", "STEP_FINISHED", "stepName"),
    ):
        opened = [event[key] for event in published if event["type"] == opener]
        closed = [event[key] for event in published if event["type"] == closer]
        assert sorted(opened) == sorted(closed), (opener, opened, closed)


def events(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[dict[str, Any]]:
    for data in publisher.data:
        EVENT.validate_json(data)  # every frame is a valid AG-UI event
    return [json.loads(data) for data in publisher.data]


async def test_a_run_publishes_its_ag_ui_events_and_ends() -> None:
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    async with publisher:
        yielded = [event async for event in arelay_agui(publisher, agent, run_input())]

    published = events(publisher)
    types = [event["type"] for event in published]
    for needed in ("TOOL_CALL_START", "TOOL_CALL_ARGS", "TOOL_CALL_END", "TOOL_CALL_RESULT", "MESSAGES_SNAPSHOT"):
        assert needed in types
    assert types[0] == "RUN_STARTED" and types[-1] == "RUN_FINISHED"
    assert_balanced(published)
    snapshot = next(event for event in published if event["type"] == "MESSAGES_SNAPSHOT")
    assert snapshot["messages"][-1]["content"] == FINAL
    if ASYNC_TOKENS:  # on 3.10 the text is in the snapshot only (see fakes.ASYNC_TOKENS)
        text = "".join(event["delta"] for event in published if event["type"] == "TEXT_MESSAGE_CONTENT")
        assert text == FIRST + FINAL
    start = next(event for event in published if event["type"] == "TOOL_CALL_START")
    assert start["toolCallName"] == "get_weather" and start["toolCallId"] == "call_1"
    arguments = "".join(event["delta"] for event in published if event["type"] == "TOOL_CALL_ARGS")
    assert json.loads(arguments) == {"city": "Turin"}
    assert next(event for event in published if event["type"] == "TOOL_CALL_RESULT")["content"] == "sunny, 21C in Turin"
    assert [event.type.value for event in yielded] == types
    assert publisher.outcome == "end"


async def test_a_failing_run_ends_with_a_masked_run_error_and_closes_fail() -> None:
    publisher = AsyncRecordingPublisher()
    async with publisher:
        async for event in arelay_agui(publisher, agent_of(failing_agent(checkpointer=InMemorySaver())), run_input()):
            if event.type.value == "RUN_ERROR":
                assert publisher.outcome == "fail", "closed as soon as the RUN_ERROR was sent"
                assert "exploded" in event.message, "the application still sees the error"
    published = events(publisher)
    assert published[-1] == {"type": "RUN_ERROR", "message": MASKED_MESSAGE}
    assert publisher.outcome == "fail"

    exposed = AsyncRecordingPublisher()
    async with exposed:
        agent = agent_of(failing_agent(checkpointer=InMemorySaver()))
        async for _ in arelay_agui(exposed, agent, run_input(), expose_errors=True):
            pass
    assert events(exposed)[-1]["message"] == "fake model exploded mid-stream"


async def run_and_cancel(publisher: AsyncRecordingPublisher, **options: Any) -> None:
    """Cancels the task while the model writes its first message (0.15 s into the model's step)."""
    agent = agent_of(weather_agent(token_delay=0.05, checkpointer=InMemorySaver()))
    model_step = asyncio.Event()

    async def run() -> None:
        async with publisher:
            async for event in arelay_agui(publisher, agent, run_input(), **options):
                if event.type.value == "STEP_STARTED":
                    model_step.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(model_step.wait(), 5)
    await asyncio.sleep(0.15)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


async def test_a_cancelled_run_finishes_cancelled_after_closing_what_is_open() -> None:
    publisher = AsyncRecordingPublisher()
    await run_and_cancel(publisher)
    published = events(publisher)
    assert published[-1] == {
        "type": "RUN_FINISHED",
        "threadId": "thread-1",
        "runId": "run-1",
        "outcome": {"type": "cancelled"},
    }
    assert published[-2] == {"type": "STEP_FINISHED", "stepName": "model"}
    if ASYNC_TOKENS:
        message_id = next(event["messageId"] for event in published if event["type"] == "TEXT_MESSAGE_START")
        assert published[-3] == {"type": "TEXT_MESSAGE_END", "messageId": message_id}
    assert_balanced(published)
    assert publisher.outcome == "fail"


async def test_a_cancelled_run_closes_end_when_asked() -> None:
    publisher = AsyncRecordingPublisher()
    await run_and_cancel(publisher, cancelled="end")
    assert events(publisher)[-1]["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "end"


async def test_a_loop_left_early_under_aclosing_is_a_cancelled_run() -> None:
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    async with publisher, contextlib.aclosing(arelay_agui(publisher, agent, run_input())) as relayed:
        async for event in relayed:
            if event.type.value == "TOOL_CALL_ARGS":
                break
    published = events(publisher)
    assert [event["type"] for event in published[-3:]] == ["TOOL_CALL_END", "STEP_FINISHED", "RUN_FINISHED"]
    assert publisher.outcome == "fail"


async def test_a_loop_left_after_the_run_finished_is_not_a_cancel() -> None:
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    async with publisher, contextlib.aclosing(arelay_agui(publisher, agent, run_input())) as relayed:
        async for event in relayed:
            if event.type.value == "RUN_FINISHED":
                break
    published = events(publisher)
    assert published[-1]["type"] == "RUN_FINISHED" and "outcome" not in published[-1]
    assert publisher.outcome == "end"


async def test_an_inert_publisher_leaves_the_run_alone() -> None:
    publisher = AsyncRecordingPublisher()
    publisher.fail()
    async with publisher:
        relayed = [
            event
            async for event in arelay_agui(
                publisher, agent_of(weather_agent(checkpointer=InMemorySaver())), run_input()
            )
        ]
    assert relayed[-1].type.value == "RUN_FINISHED"
    assert publisher.frames == []


async def test_a_synchronous_publisher_fits_too() -> None:
    publisher = RecordingPublisher()
    with publisher:
        async for _ in arelay_agui(publisher, agent_of(failing_agent(checkpointer=InMemorySaver())), run_input()):
            pass
    assert events(publisher)[-1]["type"] == "RUN_ERROR"
    assert publisher.outcome == "fail"


async def test_large_events_are_split_cut_or_dropped() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "r" * 5000

    long_text = "word " * 600
    model = FakeToolCallingModel(
        messages=iter(
            [
                AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]),  # type: ignore[list-item]
                AIMessage(content=long_text.replace(" ", "")),  # one 2,400-character chunk
            ]
        )
    )
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    reports: list[Oversize] = []
    agent = agent_of(create_agent(model, [get_weather], checkpointer=InMemorySaver()))
    async with publisher:
        async for _ in arelay_agui(publisher, agent, run_input(), on_oversize=reports.append):
            pass

    published = events(publisher)
    assert all(publisher.frame_size(data) <= 1024 for data in publisher.data)
    if ASYNC_TOKENS:  # on 3.10 the text is in the snapshot only (see fakes.ASYNC_TOKENS)
        deltas = [event for event in published if event["type"] == "TEXT_MESSAGE_CONTENT"]
        assert len(deltas) > 2, "the long delta went out as several"
        assert "".join(event["delta"] for event in deltas) == long_text.replace(" ", ""), "nothing of it lost"
    result = next(event for event in published if event["type"] == "TOOL_CALL_RESULT")
    assert result["content"].startswith("rrr") and result["content"].endswith("…[truncated]")
    assert {(report.part_type, report.truncated) for report in reports} >= {
        ("TOOL_CALL_RESULT", True),
        ("STATE_SNAPSHOT", False),
        ("MESSAGES_SNAPSHOT", False),
    }
    assert published[-1]["type"] == "RUN_FINISHED"
    assert publisher.outcome == "end"


def test_mistakes_are_refused_at_the_call() -> None:
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    with pytest.raises(ValueError, match="cancelled"):
        arelay_agui(AsyncRecordingPublisher(), agent, run_input(), cancelled="later")  # type: ignore[arg-type]
