"""The AG-UI path: the events of ag-ui-langgraph, one frame each, valid AG-UI on the wire whatever way the run ends."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

import pytest
from ag_ui.core import Event, RunAgentInput, RunFinishedEvent, RunStartedEvent, UserMessage
from ag_ui_langgraph import LangGraphAgent
from fakes import (
    ASYNC_TOKENS,
    FINAL,
    FIRST,
    USER,
    FakeToolCallingModel,
    failing_agent,
    get_weather,
    scripted_model,
    tool_call,
    weather_agent,
)
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from iq_stream_langchain import MASKED_MESSAGE, Oversize
from iq_stream_langchain.agui import arelay_agui

EVENT = TypeAdapter[Any](Event)


def run_input(thread_id: str = "thread-1", *, run_id: str = "run-1", question: str | None = None) -> RunAgentInput:
    question = question or USER["messages"][0]["content"]
    return RunAgentInput(
        thread_id=thread_id,
        run_id=run_id,
        state={},
        tools=[],
        context=[],
        forwarded_props={},
        messages=[UserMessage(id="u1", role="user", content=question)],
    )


def agent_of(graph: Any) -> LangGraphAgent:
    return LangGraphAgent(name="weather", graph=graph, emit_raw_events=False)


def assert_balanced(published: list[dict[str, Any]]) -> None:
    """Every message, tool call, step and sub-agent that was opened is closed, as AG-UI clients require of a finished
    run, by an event attributed to the same sub-agent."""
    for opener, closers, key in (
        ("TEXT_MESSAGE_START", ("TEXT_MESSAGE_END",), "messageId"),
        ("TOOL_CALL_START", ("TOOL_CALL_END",), "toolCallId"),
        ("STEP_STARTED", ("STEP_FINISHED",), "stepName"),
        ("SUBAGENT_STARTED", ("SUBAGENT_FINISHED", "SUBAGENT_ERROR"), "subagentRunId"),
    ):
        opened = sorted((event[key], event.get("subagentRunId", "")) for event in published if event["type"] == opener)
        closed = sorted((event[key], event.get("subagentRunId", "")) for event in published if event["type"] in closers)
        assert opened == closed, (opener, opened, closed)


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


async def test_with_on_error_raise_the_publishing_error_leaves_the_relay() -> None:
    from iq_stream import IQError

    publisher = AsyncRecordingPublisher(on_error="raise")
    seen = 0
    with pytest.raises(IQError):
        async with publisher:
            async for _ in arelay_agui(publisher, agent_of(weather_agent(checkpointer=InMemorySaver())), run_input()):
                seen += 1
                if seen == 2:
                    publisher.fail()
    assert seen == 2
    assert all(json.loads(data)["type"] not in ("RUN_ERROR", "RUN_FINISHED") for data in publisher.data)


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


async def test_concurrent_runs_on_one_agent_do_not_mix() -> None:
    """LangGraphAgent keeps a run's state on itself: one agent serving two requests at once needs a clone per run."""
    graph = create_agent(scripted_model(repeat=True, token_delay=0.02), [get_weather], checkpointer=InMemorySaver())
    agent = agent_of(graph)
    first, second = AsyncRecordingPublisher(), AsyncRecordingPublisher()

    async def run(publisher: AsyncRecordingPublisher, thread_id: str) -> None:
        async with publisher:
            async for _ in arelay_agui(publisher, agent, run_input(thread_id, run_id=f"run-{thread_id}")):
                pass

    await asyncio.wait_for(asyncio.gather(run(first, "a"), run(second, "b")), 20)
    for publisher, thread_id in ((first, "a"), (second, "b")):
        published = events(publisher)
        assert published[0] == {"type": "RUN_STARTED", "threadId": thread_id, "runId": f"run-{thread_id}"}
        assert published[-1]["type"] == "RUN_FINISHED" and "outcome" not in published[-1]
        assert_balanced(published)
        assert publisher.outcome == "end"


class Progress:
    """A value JSON cannot hold, in a custom event and in the state."""


class BlobState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    blob: Any


async def test_an_event_that_cannot_be_encoded_is_dropped_and_the_run_completes(
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def node(state: BlobState, config: RunnableConfig) -> dict[str, Any]:
        await adispatch_custom_event("progress", {"stage": Progress()}, config=config)
        return {"blob": b"\xff\xfe binary", "messages": [AIMessage(content="done", id="a1")]}

    graph = StateGraph(BlobState)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    publisher = AsyncRecordingPublisher()
    with caplog.at_level(logging.WARNING, logger="iq_stream_langchain"):
        async with publisher:
            relayed = [
                event.type.value
                async for event in arelay_agui(
                    publisher, agent_of(graph.compile(checkpointer=InMemorySaver())), run_input()
                )
            ]
    published = [event["type"] for event in events(publisher)]
    assert relayed[-1] == published[-1] == "RUN_FINISHED"
    assert "CUSTOM" in relayed and "CUSTOM" not in published
    assert publisher.outcome == "end"
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1 and "could not be encoded" in warnings[0]


def research_supervisor() -> Any:
    researcher = create_agent(
        FakeToolCallingModel(
            messages=iter([AIMessage(content="the researcher writes a long and slow answer here")]), token_delay=0.1
        ),
        [],
        name="researcher",
    )

    @tool
    async def research(topic: str, config: RunnableConfig) -> str:
        """Ask the researcher."""
        # The config carries the run's callbacks to the sub-agent: on Python 3.10 nothing else does.
        answer = await researcher.ainvoke({"messages": [{"role": "user", "content": topic}]}, config)
        return str(answer["messages"][-1].text)

    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("research", {"topic": "Turin"}, "c1")]), "done"]  # type: ignore[list-item]
        )
    )
    return create_agent(model, [research], checkpointer=InMemorySaver(), name="supervisor")


async def test_a_cancel_closes_the_sub_agents_that_are_open() -> None:
    agent = LangGraphAgent(
        name="supervisor", graph=research_supervisor(), emit_raw_events=False, subagent_visibility="attributed"
    )
    publisher = AsyncRecordingPublisher()
    streaming = asyncio.Event()

    async def run() -> None:
        opened = False
        async with publisher:
            async for event in arelay_agui(publisher, agent, run_input()):
                opened = opened or event.type.value == "SUBAGENT_STARTED"
                if opened and event.type.value == "TEXT_MESSAGE_CONTENT":  # the researcher writes
                    streaming.set()

    task = asyncio.create_task(run())
    await asyncio.wait_for(streaming.wait(), 10)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    published = events(publisher)
    started = [event for event in published if event["type"] == "SUBAGENT_STARTED"]
    assert started, "the run was cancelled while a sub-agent was open"
    closing = [event for event in published if event["type"] == "SUBAGENT_ERROR"]
    assert [event["subagentRunId"] for event in closing] == [event["subagentRunId"] for event in started]
    assert closing[-1] == {
        "type": "SUBAGENT_ERROR",
        "subagentRunId": started[-1]["subagentRunId"],
        "message": "run cancelled",
        "code": "cancelled",
    }
    assert published[-1]["outcome"] == {"type": "cancelled"}
    assert_balanced(published)
    assert publisher.outcome == "fail"


class ScriptedAgent:
    """Stands in for a LangGraphAgent: yields the given events, then raises the given error."""

    def __init__(self, script: list[Any], error: BaseException | None = None) -> None:
        self.script = script
        self.error = error

    def clone(self) -> ScriptedAgent:
        return self

    async def run(self, run_input: RunAgentInput) -> AsyncIterator[Any]:
        for event in self.script:
            yield event
        if self.error is not None:
            raise self.error


async def test_a_run_that_raises_past_ag_ui_langgraph_ends_with_its_run_error() -> None:
    for expose, message in ((False, MASKED_MESSAGE), (True, "the API key sk-123 was refused")):
        publisher = AsyncRecordingPublisher()
        agent = cast("Any", ScriptedAgent([], ValueError("the API key sk-123 was refused")))
        with pytest.raises(ValueError):
            async with publisher:
                async for _ in arelay_agui(publisher, agent, run_input(), expose_errors=expose):
                    pass
        assert events(publisher) == [
            {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"},
            {"type": "RUN_ERROR", "message": message},
        ]
        assert publisher.outcome == "fail"


async def test_a_run_event_over_the_limit_goes_without_its_optional_field() -> None:
    script = [
        RunStartedEvent(thread_id="thread-1", run_id="run-1", input=run_input(question="q" * 3000)),
        RunFinishedEvent(thread_id="thread-1", run_id="run-1", result={"answer": "a" * 3000}),
    ]
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    reports: list[Oversize] = []
    async with publisher:
        async for _ in arelay_agui(
            publisher, cast("Any", ScriptedAgent(script)), run_input(), on_oversize=reports.append
        ):
            pass
    assert events(publisher) == [
        {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"},
        {"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-1"},
    ]
    assert [(report.part_type, report.truncated) for report in reports] == [
        ("RUN_STARTED", True),
        ("RUN_FINISHED", True),
    ]


async def test_an_on_oversize_hook_that_raises_never_breaks_the_run() -> None:
    def broken(report: Oversize) -> None:
        raise ConnectionError("statsd is down")

    script = [
        RunStartedEvent(thread_id="thread-1", run_id="run-1"),
        RunFinishedEvent(thread_id="thread-1", run_id="run-1", result={"answer": "a" * 3000}),
    ]
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    async with publisher:
        async for _ in arelay_agui(publisher, cast("Any", ScriptedAgent(script)), run_input(), on_oversize=broken):
            pass
    assert [event["type"] for event in events(publisher)] == ["RUN_STARTED", "RUN_FINISHED"]
    assert publisher.outcome == "end"


async def test_an_exception_of_the_callers_loop_never_closes_end() -> None:
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    with pytest.raises(KeyError):
        async with (
            publisher,
            contextlib.aclosing(arelay_agui(publisher, agent, run_input(), cancelled="end")) as relayed,
        ):
            async for event in relayed:
                if event.type.value == "TOOL_CALL_RESULT":
                    raise KeyError("the application's own bug")
    assert events(publisher)[-1]["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "fail"
