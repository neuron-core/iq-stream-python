"""The AG-UI path: the events of ag-ui-langgraph, one frame each, valid AG-UI on the wire whatever way the run ends."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import sys
import time
from collections.abc import AsyncIterator
from typing import Annotated, Any, cast

import pytest
from ag_ui.core import Event, RunAgentInput, RunErrorEvent, RunFinishedEvent, RunStartedEvent, UserMessage
from ag_ui_langgraph import LangGraphAgent
from fakes import (
    ASYNC_TOKENS,
    FINAL,
    FIRST,
    USER,
    FakeToolCallingModel,
    failing_agent,
    get_weather,
    hitl_agent,
    research_supervisor,
    scripted_model,
    stubborn_agent,
    tool_call,
    weather_agent,
)
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain.agents import create_agent
from langchain_core.callbacks import adispatch_custom_event
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import TypeAdapter
from typing_extensions import TypedDict

from iq_stream_langchain import MASKED_MESSAGE, Oversize
from iq_stream_langchain.agui import arelay_agui

EVENT = TypeAdapter[Any](Event)
CANCELLING = sys.version_info >= (3, 11)


class TimedPublisher(AsyncRecordingPublisher):
    """Notes when the stream was first closed."""

    closed_at: float | None = None

    async def close(self, outcome: Any = "end", timeout: float = 10.0) -> Any:
        if self.closed_at is None:
            self.closed_at = time.monotonic()
        return await super().close(outcome, timeout)


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
    await run_and_cancel(publisher, close_on_cancel="end")
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


async def test_a_loop_left_at_run_finished_is_a_cancelled_run() -> None:
    """RUN_FINISHED goes out once the loop is over and the answer saved; a loop left at it never got there."""
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    async with publisher, contextlib.aclosing(arelay_agui(publisher, agent, run_input())) as relayed:
        async for event in relayed:
            if event.type.value == "RUN_FINISHED":
                break
    published = events(publisher)
    assert [event["type"] for event in published].count("RUN_FINISHED") == 1
    assert published[-1]["outcome"] == {"type": "cancelled"}
    assert_balanced(published)
    assert publisher.outcome == "fail"


async def test_run_finished_waits_for_the_save() -> None:
    publisher = AsyncRecordingPublisher()
    seen_at_save: list[str] = []

    async def save(finished: RunFinishedEvent) -> None:
        await asyncio.sleep(0)
        seen_at_save.extend(json.loads(data)["type"] for data in publisher.data)
        assert finished.thread_id == "thread-1" and publisher.outcome is None

    agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
    async with publisher:
        yielded = [event.type.value async for event in arelay_agui(publisher, agent, run_input(), before_close=save)]
    assert yielded[-1] == "RUN_FINISHED", "the application sees it when it comes"
    assert "RUN_FINISHED" not in seen_at_save and seen_at_save[-1] == "MESSAGES_SNAPSHOT"
    published = events(publisher)
    assert published[-1]["type"] == "RUN_FINISHED" and "outcome" not in published[-1]
    assert publisher.outcome == "end"


class SaveFailed(Exception):
    pass


async def test_a_failed_save_ends_the_run_with_a_run_error_not_run_finished() -> None:
    """StreamAgent takes RUN_FINISHED as the end and ignores the iq:fail after it: a failed save must not follow it."""

    def save(finished: RunFinishedEvent) -> None:
        raise SaveFailed("the database at 10.0.0.1 is down")

    for expose, message in ((False, MASKED_MESSAGE), (True, "the database at 10.0.0.1 is down")):
        publisher = AsyncRecordingPublisher()
        agent = agent_of(weather_agent(checkpointer=InMemorySaver()))
        with pytest.raises(SaveFailed):
            async with publisher:
                async for _ in arelay_agui(publisher, agent, run_input(), before_close=save, expose_errors=expose):
                    pass
        published = events(publisher)
        assert "RUN_FINISHED" not in [event["type"] for event in published]
        assert published[-1] == {"type": "RUN_ERROR", "message": message}
        assert_balanced(published)
        assert publisher.outcome == "fail"


async def test_a_save_on_an_inert_publisher_still_runs() -> None:
    publisher = AsyncRecordingPublisher()
    publisher.fail()
    saved: list[str] = []
    async with publisher:
        async for _ in arelay_agui(
            publisher,
            agent_of(weather_agent(checkpointer=InMemorySaver())),
            run_input(),
            before_close=lambda finished: saved.append(finished.run_id),
        ):
            pass
    assert saved == ["run-1"]
    assert publisher.frames == []


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
    with pytest.raises(ValueError, match="close_on_cancel"):
        arelay_agui(AsyncRecordingPublisher(), agent, run_input(), close_on_cancel="later")  # type: ignore[arg-type]


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
            contextlib.aclosing(arelay_agui(publisher, agent, run_input(), close_on_cancel="end")) as relayed,
        ):
            async for event in relayed:
                if event.type.value == "TOOL_CALL_RESULT":
                    raise KeyError("the application's own bug")
    assert events(publisher)[-1]["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "fail"


async def hitl_run(publisher: AsyncRecordingPublisher, **options: Any) -> list[str]:
    """A run that waits for a human to approve a long email; returns what the application saw."""
    agent = LangGraphAgent(name="mail", graph=hitl_agent(), emit_raw_events=False, **options)
    async with publisher:
        return [event.type.value async for event in arelay_agui(publisher, agent, run_input())]


async def test_an_interrupt_over_the_limit_still_asks_the_human() -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    seen = await hitl_run(publisher)
    published = events(publisher)
    assert seen[-2:] == ["CUSTOM", "RUN_FINISHED"]
    asking = next(event for event in published if event["type"] == "CUSTOM" and event["name"] == "on_interrupt")
    value = json.loads(asking["value"])  # JSON text, as ag_ui_langgraph writes it
    [request] = value["action_requests"]
    assert request["name"] == "send_email" and request["args"]["to"] == "team@example.com"
    assert request["args"]["body"].startswith("Dear team") and request["args"]["body"].endswith("…[truncated]")
    assert value["review_configs"][0]["action_name"] == "send_email"
    assert published[-1]["type"] == "RUN_FINISHED"
    assert all(publisher.frame_size(data) <= 1024 for data in publisher.data)
    assert publisher.outcome == "end"


async def test_an_interrupt_outcome_over_the_limit_keeps_run_finished_and_its_ids() -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    reports: list[Oversize] = []
    agent = LangGraphAgent(name="mail", graph=hitl_agent(), emit_raw_events=False, emit_interrupt_outcome=True)
    async with publisher:
        async for _ in arelay_agui(publisher, agent, run_input(), on_oversize=reports.append):
            pass
    published = events(publisher)
    finished = published[-1]
    assert finished["type"] == "RUN_FINISHED" and finished["outcome"]["type"] == "interrupt"
    [interrupt] = finished["outcome"]["interrupts"]
    state = await agent.graph.aget_state({"configurable": {"thread_id": "thread-1"}})
    assert interrupt["id"] == state.interrupts[0].id, "the id the resume answers"
    assert interrupt["metadata"]["langgraph"]["raw"]["action_requests"][0]["name"] == "send_email"
    assert ("RUN_FINISHED", True) in {(report.part_type, report.truncated) for report in reports}
    assert all(publisher.frame_size(data) <= 1024 for data in publisher.data)


async def test_a_run_finished_that_cannot_be_cut_goes_without_its_outcome() -> None:
    interrupts = [{"id": f"interrupt-{index:04d}", "reason": "langgraph:interrupt"} for index in range(40)]
    script = [
        RunStartedEvent(thread_id="thread-1", run_id="run-1"),
        RunFinishedEvent.model_validate(
            {"threadId": "thread-1", "runId": "run-1", "outcome": {"type": "interrupt", "interrupts": interrupts}}
        ),
    ]
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    async with publisher:
        async for _ in arelay_agui(publisher, cast("Any", ScriptedAgent(script)), run_input()):
            pass
    assert events(publisher)[-1] == {"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-1"}


async def test_an_exposed_error_over_the_limit_goes_masked() -> None:
    long = "the provider refused the request: " + "x" * 2000
    script = [RunStartedEvent(thread_id="thread-1", run_id="run-1"), RunErrorEvent(message=long, code="provider")]
    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    async with publisher:
        async for _ in arelay_agui(publisher, cast("Any", ScriptedAgent(script)), run_input(), expose_errors=True):
            pass
    assert events(publisher)[-1] == {"type": "RUN_ERROR", "message": MASKED_MESSAGE, "code": "provider"}
    assert publisher.outcome == "fail"

    publisher = AsyncRecordingPublisher(max_frame_bytes=1024)
    with pytest.raises(ValueError):
        async with publisher:
            agent = cast("Any", ScriptedAgent([], ValueError(long)))
            async for _ in arelay_agui(publisher, agent, run_input(), expose_errors=True):
                pass
    assert events(publisher)[-1] == {"type": "RUN_ERROR", "message": MASKED_MESSAGE}
    assert publisher.outcome == "fail"


async def test_a_failed_tool_result_is_masked_in_the_snapshots() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        raise ToolException("the API key sk-123 was refused")

    get_weather.handle_tool_error = True
    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sorry."])  # type: ignore[list-item]
    )
    publisher = AsyncRecordingPublisher()
    async with publisher:
        async for _ in arelay_agui(
            publisher, agent_of(create_agent(model, [get_weather], checkpointer=InMemorySaver())), run_input()
        ):
            pass
    snapshots = [event for event in events(publisher) if event["type"] in ("MESSAGES_SNAPSHOT", "STATE_SNAPSHOT")]
    assert snapshots and not any("sk-123" in json.dumps(event) for event in snapshots)
    tool_messages = [message for message in snapshots[-1]["messages"] if message["role"] == "tool"]
    assert tool_messages == [
        {"id": tool_messages[0]["id"], "role": "tool", "content": MASKED_MESSAGE, "toolCallId": "c1", "error": "error"}
    ]


@pytest.mark.skipif(not CANCELLING, reason="Python 3.10 has no Task.cancelling(): a cancel in the loop body is a break")
async def test_a_cancel_that_lands_in_the_loop_body_closes_with_close_on_cancel() -> None:
    publisher = AsyncRecordingPublisher()
    agent = agent_of(weather_agent(token_delay=0.02, checkpointer=InMemorySaver()))
    streaming = asyncio.Event()

    async def run() -> None:
        async with (
            publisher,
            contextlib.aclosing(arelay_agui(publisher, agent, run_input(), close_on_cancel="end")) as relayed,
        ):
            async for event in relayed:
                if event.type.value == "TEXT_MESSAGE_CONTENT":
                    streaming.set()
                await asyncio.sleep(0.05)  # the application's own wait: a websocket send, a save

    task = asyncio.create_task(run())
    await asyncio.wait_for(streaming.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert events(publisher)[-1]["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "end"


async def test_a_cancel_is_published_before_the_graph_has_stopped() -> None:
    """A tool that finishes its work before it lets a cancel through holds LangGraph up; the page hears at once."""
    for stop in ("break", "cancel"):
        publisher = TimedPublisher()
        entered = asyncio.Event()
        relayed = arelay_agui(publisher, agent_of(stubborn_agent(1.0, entered)), run_input())

        async def run(
            publisher: TimedPublisher = publisher, relayed: Any = relayed, leave: bool = stop == "break"
        ) -> None:
            async with publisher, contextlib.aclosing(relayed):
                async for event in relayed:
                    if leave and event.type.value == "STEP_STARTED" and event.step_name == "tools":
                        await entered.wait()  # noqa: B023
                        break

        task = asyncio.create_task(run())
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.sleep(0.1)
        stopped = time.monotonic()
        if stop == "cancel":
            task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        assert publisher.closed_at is not None and publisher.closed_at - stopped < 0.5, stop
        assert time.monotonic() - stopped > 0.8, f"{stop}: the graph stopped once the tool was done"
        assert events(publisher)[-1]["outcome"] == {"type": "cancelled"}
        assert publisher.outcome == "fail"
