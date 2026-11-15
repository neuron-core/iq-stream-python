"""The tee with Agno's own event JSON: what is published, what the caller gets, and what goes wrong safely."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.models.response import ToolExecution
from agno.run.agent import (
    CustomEvent,
    RunCompletedEvent,
    RunContentEvent,
    RunErrorEvent,
    RunOutput,
    RunStartedEvent,
    ToolCallCompletedEvent,
)
from agno.run.base import BaseRunOutputEvent
from agno.run.team import RunCompletedEvent as TeamRunCompletedEvent
from agno.run.team import RunContentEvent as TeamRunContentEvent
from agno.run.team import RunStartedEvent as TeamRunStartedEvent
from agno.team import Team
from agno.workflow import Step, Workflow
from agno.workflow.types import StepInput, StepOutput
from iq_stream import FrameTooLarge, IQError
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from recorded import Errors, collect, names, payloads
from scripted_model import BIG, ScriptedModel, broken_tool, weather_agent

from iq_stream_agno import MASKED_ERROR, RunSummary, apublish, publish
from iq_stream_agno._native import dumps


def dicts(events: list[Any]) -> list[dict[str, Any]]:
    """The events as the browser should read them: to_dict(), through JSON."""
    return [json.loads(dumps(event.to_dict())) for event in events if isinstance(event, BaseRunOutputEvent)]


async def test_an_async_run_is_published_event_for_event_and_ends() -> None:
    publisher = AsyncRecordingPublisher()
    agent = weather_agent()
    received = await collect(apublish(agent.arun("Weather?", stream=True, stream_events=True), publisher))
    assert payloads(publisher) == dicts(received)
    assert [frame.event for frame in publisher.frames] == [None] * len(received)
    assert "ToolCallStarted" in names(publisher) and names(publisher)[-1] == "RunCompleted"
    assert publisher.outcome == "end" and publisher.close_calls == 1


def test_a_sync_run_is_published_the_same_way() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent()
    received = list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher))
    assert payloads(publisher) == dicts(received)
    tool = next(payload for payload in payloads(publisher) if payload["event"] == "ToolCallCompleted")["tool"]
    assert tool["tool_name"] == "get_weather" and tool["tool_args"] == {"city": "Paris"}
    assert tool["result"] == "21C and sunny in Paris"
    assert publisher.outcome == "end"


def test_without_stream_events_only_the_content_is_published() -> None:
    publisher = RecordingPublisher()
    received = list(publish(weather_agent().run("Weather?", stream=True), publisher))
    assert set(names(publisher)) == {"RunContent"}
    assert len(publisher.frames) == len(received)
    assert "".join(payload.get("content") or "" for payload in payloads(publisher)).endswith("sunny in Paris.")
    assert publisher.outcome == "end"


@dataclass
class Progress(CustomEvent):
    """A custom event as Agno's documentation writes one: a dataclass subclass."""

    event: str = "Progress"
    percent: int = 0


def test_event_names_name_each_frame_after_its_event_and_leave_odd_names_unnamed() -> None:
    publisher = RecordingPublisher()
    events = [
        RunStartedEvent(run_id="r1", session_id="s1"),
        Progress(event="not a valid name!", run_id="r1"),
        Progress(run_id="r1", percent=50),
        RunCompletedEvent(run_id="r1", content="done"),
    ]
    list(publish(iter(events), publisher, event_names=True))
    assert [frame.event for frame in publisher.frames] == ["RunStarted", None, "Progress", "RunCompleted"]
    assert names(publisher) == ["RunStarted", "not a valid name!", "Progress", "RunCompleted"]


def test_a_failed_run_is_not_raised_ends_with_run_error_and_fails() -> None:
    publisher = RecordingPublisher()
    received = list(publish(weather_agent(fail_on_turn=2).run("Weather?", stream=True, stream_events=True), publisher))
    assert received[-1].event == "RunError"
    last = payloads(publisher)[-1]
    assert last["event"] == "RunError" and last["content"] == MASKED_ERROR and last["error_type"] == "RuntimeError"
    assert "scripted model failure" not in "".join(publisher.data)
    assert publisher.outcome == "fail"


def test_expose_errors_publishes_the_error_text() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent(fail_on_turn=2)
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, expose_errors=True))
    assert payloads(publisher)[-1]["content"] == "scripted model failure"


def test_a_failing_tool_s_error_text_is_masked() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent(tool_name="broken_tool")
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher))
    assert "secret database password" not in "".join(publisher.data)
    failed = [payload for payload in payloads(publisher) if payload["event"] in ("ToolCallCompleted", "ToolCallError")]
    assert failed and all(payload["tool"]["result"] == MASKED_ERROR for payload in failed)
    assert publisher.outcome == "end"  # the model answered after the tool's error: the run completed


def test_an_iterator_that_raises_reaches_the_caller_unchanged_and_fails_the_stream() -> None:
    publisher = RecordingPublisher()
    problem = RuntimeError("iterator blew up")

    def events() -> Iterator[Any]:
        for index, event in enumerate(weather_agent().run("Weather?", stream=True, stream_events=True)):
            if index == 3:
                raise problem
            yield event

    with pytest.raises(RuntimeError) as raised:
        list(publish(events(), publisher))
    assert raised.value is problem
    assert len(publisher.frames) == 3 and publisher.outcome == "fail"


def test_include_exclude_and_a_transform_that_returns_none() -> None:
    agent = weather_agent()
    only = RecordingPublisher()
    list(publish(agent.run("Weather?", stream=True, stream_events=True), only, include={"ToolCallStarted"}))
    assert names(only) == ["ToolCallStarted"]

    without = RecordingPublisher()
    exclude = ("ModelRequestStarted", "ModelRequestCompleted")
    list(publish(agent.run("Weather?", stream=True, stream_events=True), without, exclude=exclude))
    assert not set(exclude) & set(names(without)) and names(without)[-1] == "RunCompleted"

    def slim(event: BaseRunOutputEvent) -> dict[str, Any] | None:
        if not isinstance(event, RunContentEvent):
            return None
        return {"event": "RunContent", "delta": event.content}

    transformed = RecordingPublisher()
    list(publish(agent.run("Weather?", stream=True, stream_events=True), transformed, transform=slim))
    assert {tuple(payload) for payload in payloads(transformed)} == {("event", "delta")}
    assert without.outcome == only.outcome == transformed.outcome == "end"


def test_an_excluded_terminal_event_still_decides_the_close_line() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent(fail_on_turn=2)
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, exclude={"RunError"}))
    assert "RunError" not in names(publisher) and publisher.outcome == "fail"


def test_the_run_output_yielded_last_is_passed_through_and_not_published() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent()
    received = list(publish(agent.run("Weather?", stream=True, stream_events=True, yield_run_output=True), publisher))
    assert isinstance(received[-1], RunOutput)
    assert len(publisher.frames) == len(received) - 1 and names(publisher)[-1] == "RunCompleted"


def test_an_oversize_tool_result_is_cut_and_marked_and_the_line_fits() -> None:
    BIG["size"] = 5_000
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    errors = Errors()
    agent = weather_agent(tool_name="big_report")
    received = list(publish(agent.run("Report?", stream=True, stream_events=True), publisher, on_dropped=errors))
    assert len(publisher.frames) == len([event for event in received if isinstance(event, BaseRunOutputEvent)])
    assert all(publisher.frame_size(frame.data) <= 2_048 for frame in publisher.frames)
    completed = next(payload for payload in payloads(publisher) if payload["event"] == "ToolCallCompleted")
    assert completed["iq_truncated"] == ["tool.result"]
    assert 0 < len(completed["tool"]["result"]) < 5_000 and set(completed["tool"]["result"]) == {"x"}
    assert names(publisher)[-1] == "RunCompleted" and "iq_truncated" not in payloads(publisher)[-1]
    assert errors == [] and publisher.outcome == "end"


def test_tool_results_are_cut_before_the_answer() -> None:
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    tools = [ToolExecution(tool_call_id=f"call_{n}", tool_name="big_report", result="x" * 1_500) for n in (1, 2)]
    event = RunCompletedEvent(run_id="r1", content="The reports are in.", tools=tools)
    list(publish(iter([event]), publisher))
    (payload,) = payloads(publisher)
    assert payload["iq_truncated"] == ["tools[].result"] and payload["content"] == "The reports are in."
    assert publisher.frame_size(publisher.data[0]) <= 2_048


def test_the_shrink_policy_removes_nested_runs_first() -> None:
    publisher = RecordingPublisher(max_frame_bytes=1_024)
    member = RunOutput(run_id="member", content="m" * 3_000)
    event = TeamRunCompletedEvent(
        run_id="team",
        content="The member says it is sunny.",
        member_responses=[member],
        images=[],  # media and context too: removing the nested runs alone is enough, so they stay
        session_state={"city": "Paris"},
    )
    list(publish(iter([event]), publisher))
    (payload,) = payloads(publisher)
    assert payload["iq_truncated"] == ["member_responses"] and "member_responses" not in payload
    assert payload["content"] == "The member says it is sunny."
    assert payload["images"] == [] and payload["session_state"] == {"city": "Paris"}


def test_oversize_tool_arguments_are_removed_and_the_tool_call_still_published() -> None:
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    errors = Errors()
    agent = weather_agent(tool_args={"city": "P" * 3_000})
    received = list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, on_dropped=errors))
    assert len(publisher.frames) == len(received) and errors == []
    assert all(publisher.frame_size(data) <= 2_048 for data in publisher.data)
    started = next(payload for payload in payloads(publisher) if payload["event"] == "ToolCallStarted")
    assert started["iq_truncated"] == ["tool.tool_args"] and "tool_args" not in started["tool"]
    assert started["tool"]["tool_name"] == "get_weather" and started["tool"]["tool_call_id"] == "call_1"
    completed = next(payload for payload in payloads(publisher) if payload["event"] == "ToolCallCompleted")
    assert completed["iq_truncated"] == ["tool.result", "tool.tool_args"]
    assert completed["tool"]["result"].startswith("21C and sunny in PPP")


def test_a_small_result_is_kept_whole_when_the_arguments_are_what_does_not_fit() -> None:
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    tool = ToolExecution(tool_call_id="call_1", tool_name="write_file", tool_args={"text": "x" * 3_000}, result="ok")
    list(publish(iter([ToolCallCompletedEvent(run_id="r1", content="write_file completed.", tool=tool)]), publisher))
    (payload,) = payloads(publisher)
    assert payload["iq_truncated"] == ["tool.tool_args"] and payload["tool"]["result"] == "ok"
    assert payload["content"] == "write_file completed."


def test_the_tools_listed_in_run_completed_lose_their_arguments_before_the_answer_is_cut() -> None:
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    tools = [ToolExecution(tool_call_id="call_1", tool_name="write_file", tool_args={"text": "x" * 3_000}, result="ok")]
    list(publish(iter([RunCompletedEvent(run_id="r1", content="The file is written.", tools=tools)]), publisher))
    (payload,) = payloads(publisher)
    assert payload["iq_truncated"] == ["tools[].tool_args"] and payload["content"] == "The file is written."
    assert payload["tools"][0]["result"] == "ok" and "tool_args" not in payload["tools"][0]


def test_an_error_event_s_additional_data_is_not_published() -> None:
    publisher = RecordingPublisher()
    event = RunErrorEvent(run_id="r1", content="connection to db:5432 refused", additional_data={"dsn": "secret"})
    list(publish(iter([event]), publisher))
    (payload,) = payloads(publisher)
    assert payload["content"] == MASKED_ERROR and "additional_data" not in payload and "secret" not in publisher.data[0]
    assert publisher.outcome == "fail"


def test_with_on_oversize_drop_an_oversize_frame_is_dropped_and_reported() -> None:
    BIG["size"] = 5_000
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    errors = Errors()
    agent = weather_agent(tool_name="big_report")
    stream = agent.run("Report?", stream=True, stream_events=True)
    list(publish(stream, publisher, oversize="drop", on_dropped=errors))
    assert "ToolCallCompleted" not in names(publisher) and names(publisher)[-1] == "RunCompleted"
    assert len(errors) == 1 and isinstance(errors[0], FrameTooLarge) and errors[0].limit == 2_048
    assert publisher.outcome == "end"


def test_an_oversize_frame_is_logged_once_without_its_data_by_default(caplog: pytest.LogCaptureFixture) -> None:
    BIG["size"] = 5_000
    publisher = RecordingPublisher(max_frame_bytes=2_048)
    agent = weather_agent(tool_name="big_report")
    list(publish(agent.run("Report?", stream=True, stream_events=True), publisher, oversize="drop"))
    warnings = [record for record in caplog.records if record.name == "iq_stream_agno"]
    assert len(warnings) == 1 and "xxxx" not in warnings[0].getMessage()


def test_a_publisher_that_gave_up_stops_the_encoding_and_the_run_goes_on() -> None:
    publisher = RecordingPublisher()
    publisher.fail()
    encoded: list[str] = []

    def spy(event: BaseRunOutputEvent) -> dict[str, Any]:
        encoded.append(str(event.event))
        return event.to_dict()

    received = list(publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher, transform=spy))
    assert received[-1].event == "RunCompleted" and encoded == [] and publisher.frames == []


def test_a_publisher_that_gives_up_midway_does_not_break_the_run() -> None:
    publisher = RecordingPublisher()
    received: list[Any] = []
    for event in publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher):
        received.append(event)
        if event.event == "ToolCallStarted":
            publisher.fail()
    assert received[-1].event == "RunCompleted" and names(publisher)[-1] == "ToolCallStarted"


def test_with_on_error_raise_the_publisher_s_error_reaches_the_caller() -> None:
    publisher = RecordingPublisher(on_error="raise")
    publisher.fail()
    with pytest.raises(IQError):
        list(publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher))


def test_an_encoder_that_raises_is_reported_once_and_the_run_is_unaffected() -> None:
    publisher = RecordingPublisher()
    errors = Errors()

    def flaky(event: BaseRunOutputEvent) -> dict[str, Any]:
        if event.event == "ToolCallStarted":
            raise ValueError("cannot encode")
        return event.to_dict()

    received = list(
        publish(
            weather_agent().run("Weather?", stream=True, stream_events=True),
            publisher,
            transform=flaky,
            on_dropped=errors,
        )
    )
    assert [error.code for error in errors] == ["encode_failed"]
    assert isinstance(errors[0].__cause__, ValueError) and "ToolCallStarted" in str(errors[0])
    assert received[-1].event == "RunCompleted" and "ToolCallStarted" not in names(publisher)


def test_a_source_of_strings_is_passed_through_reported_and_fails_the_stream() -> None:
    publisher = RecordingPublisher()
    errors = Errors()
    received = list(publish(iter(["event: RunStarted\ndata: {}\n\n", "data: {}\n\n"]), publisher, on_dropped=errors))
    assert len(received) == 2 and publisher.frames == []
    assert [error.code for error in errors] == ["unsupported_source"] and publisher.outcome == "fail"


def test_a_run_output_instead_of_an_iterator_closes_the_stream_and_says_why() -> None:
    publisher = RecordingPublisher()
    output = weather_agent().run("Weather?")  # stream=True forgotten
    with pytest.raises(TypeError, match=r"stream=True"):
        list(publish(output, publisher))  # type: ignore[arg-type]
    assert publisher.outcome == "fail" and publisher.close_calls == 1


async def test_a_coroutine_instead_of_an_async_iterator_closes_the_stream_and_is_not_left_unawaited() -> None:
    publisher = AsyncRecordingPublisher()
    coroutine = weather_agent().arun("Weather?")  # stream=True forgotten
    with pytest.raises(TypeError, match=r"stream=True"):
        await collect(apublish(coroutine, publisher))  # type: ignore[arg-type]
    assert publisher.outcome == "fail" and publisher.close_calls == 1
    assert inspect.getcoroutinestate(coroutine) == inspect.CORO_CLOSED  # no "never awaited" warning


def test_an_async_publisher_is_refused_by_publish() -> None:
    with pytest.raises(TypeError):
        publish(iter([]), AsyncRecordingPublisher())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "options",
    [
        {"protocol": "html"},
        {"close_on_cancel": "maybe"},
        {"oversize": "split"},
        {"protocol": "ag-ui", "transform": dict},
    ],
)
def test_invalid_options_are_refused_at_the_call(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        publish(iter([]), RecordingPublisher(), **options)


def test_before_close_sees_the_summary_before_the_close_line() -> None:
    order: list[str] = []
    publisher = RecordingPublisher()

    def before_close(summary: RunSummary) -> None:
        order.append(f"before_close {summary.status} {summary.outcome} {publisher.outcome}")

    agent = weather_agent()
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, before_close=before_close))
    assert order == ["before_close completed end None"] and publisher.outcome == "end"


def test_before_close_is_not_called_when_the_iterator_raised() -> None:
    publisher = RecordingPublisher()
    called: list[RunSummary] = []

    def events() -> Iterator[Any]:
        yield from weather_agent().run("Weather?", stream=True)
        raise RuntimeError("after the last event")

    with pytest.raises(RuntimeError):
        list(publish(events(), publisher, before_close=called.append))
    assert called == [] and publisher.outcome == "fail"


def test_a_before_close_that_raises_fails_the_stream_and_reaches_the_caller() -> None:
    publisher = RecordingPublisher()

    def before_close(summary: RunSummary) -> None:
        raise OSError("database down")

    with pytest.raises(OSError):
        list(publish(weather_agent().run("Weather?", stream=True), publisher, before_close=before_close))
    assert publisher.outcome == "fail"


def test_tool_completed_events_without_a_tool_still_encode() -> None:
    publisher = RecordingPublisher()
    list(publish(iter([ToolCallCompletedEvent(run_id="r1", content="done")]), publisher))
    assert names(publisher) == ["ToolCallCompleted"] and publisher.outcome == "end"


SECRET = "secret database password"


def broken_team() -> Team:
    member = Agent(
        id="member",
        name="Member",
        role="weather",
        model=ScriptedModel(tool_name="broken_tool"),
        tools=[broken_tool],
        telemetry=False,
    )
    leader = ScriptedModel(
        tool_name="delegate_task_to_member",
        tool_args={"member_id": "member", "task": "weather in Paris"},
        answer="The member says it is sunny.",
    )
    return Team(id="team", name="Team", model=leader, members=[member], telemetry=False)


def broken_workflow() -> Workflow:
    steps = [Step(name="lookup", agent=weather_agent(tool_name="broken_tool"))]
    return Workflow(id="wf", name="WF", steps=steps, telemetry=False)


@pytest.mark.parametrize(
    ("make", "ending"), [(broken_team, "TeamRunCompleted"), (broken_workflow, "WorkflowCompleted")]
)
async def test_a_nested_run_s_failed_tool_is_masked_in_the_events_that_carry_it(make: Any, ending: str) -> None:
    publisher = AsyncRecordingPublisher()
    await collect(apublish(make().arun("Weather?", stream=True, stream_events=True), publisher))
    assert SECRET not in "".join(publisher.data) and names(publisher)[-1] == ending
    nested = payloads(publisher)[-1].get("member_responses") or payloads(publisher)[-1]["step_executor_runs"]
    (tool,) = nested[0]["tools"]
    assert tool["tool_call_error"] is True and tool["result"] == MASKED_ERROR
    exposed = AsyncRecordingPublisher()
    await collect(apublish(make().arun("Weather?", stream=True, stream_events=True), exposed, expose_errors=True))
    assert SECRET in exposed.data[-1]


def test_a_failed_step_s_error_is_masked_in_the_workflow_s_ending() -> None:
    def boom(step_input: StepInput) -> StepOutput:
        raise RuntimeError(SECRET)

    publisher = RecordingPublisher()
    workflow = Workflow(id="wf", name="WF", steps=[Step(name="boom", executor=boom)], telemetry=False)
    list(publish(workflow.run("Go", stream=True, stream_events=True), publisher))
    (result,) = payloads(publisher)[-1]["step_results"]
    assert result["success"] is False and result["error"] == MASKED_ERROR


@pytest.mark.parametrize(
    ("size", "removed"), [(1_500, ["requirements"]), (5_000, ["requirements", "tools[].tool_args"])]
)
def test_a_pause_with_large_arguments_is_published_without_its_requirements(size: int, removed: list[str]) -> None:
    publisher = RecordingPublisher(max_frame_bytes=4_096)
    errors = Errors()
    agent = weather_agent(tool_name="book_trip", tool_args={"city": "P" * size}, db=InMemoryDb())
    list(publish(agent.run("Book", stream=True, stream_events=True), publisher, on_dropped=errors))
    paused = payloads(publisher)[-1]
    assert paused["event"] == "RunPaused" and paused["iq_truncated"] == removed and errors == []
    assert paused["tools"][0]["tool_name"] == "book_trip" and ("tool_args" in paused["tools"][0]) == (size == 1_500)
    assert publisher.outcome == "end"


def test_a_short_tool_result_is_kept_while_the_longer_summary_is_cut() -> None:
    publisher = RecordingPublisher(max_frame_bytes=4_096)
    tool = ToolExecution(tool_call_id="call_1", tool_name="write_file", tool_args={"text": "x" * 2_500}, result="ok")
    summary = f"write_file(text={'x' * 2_500}) completed in 0.1s."
    list(publish(iter([ToolCallCompletedEvent(run_id="r1", content=summary, tool=tool)]), publisher))
    (payload,) = payloads(publisher)
    assert payload["iq_truncated"] == ["content"] and payload["tool"]["result"] == "ok"
    assert "tool_args" in payload["tool"] and summary.startswith(payload["content"])


def ending_events(kind: str = "agent") -> list[Any]:
    started, content, completed = {
        "agent": (RunStartedEvent, RunContentEvent, RunCompletedEvent),
        "team": (TeamRunStartedEvent, TeamRunContentEvent, TeamRunCompletedEvent),
    }[kind]
    return [started(run_id="r1"), content(run_id="r1", content="Hi"), completed(run_id="r1", content="Hi")]


@pytest.mark.parametrize(("kind", "error"), [("agent", "RunError"), ("team", "TeamRunError")])
def test_a_failed_save_replaces_the_held_ending_with_a_masked_error(kind: str, error: str) -> None:
    publisher = RecordingPublisher()

    def before_close(summary: RunSummary) -> None:
        raise OSError("database down at db.internal:5432")

    with pytest.raises(OSError):
        list(publish(iter(ending_events(kind)), publisher, before_close=before_close))
    assert names(publisher) == [*names(publisher)[:2], error] and names(publisher)[0].endswith("RunStarted")
    assert payloads(publisher)[-1]["content"] == MASKED_ERROR and payloads(publisher)[-1]["run_id"] == "r1"
    assert "Completed" not in "".join(names(publisher)) and publisher.outcome == "fail"


def test_the_ending_goes_out_after_a_save_that_succeeded() -> None:
    publisher = RecordingPublisher()
    seen: list[list[str]] = []

    def before_close(summary: RunSummary) -> None:
        seen.append(names(publisher))

    list(publish(iter(ending_events()), publisher, before_close=before_close))
    assert seen == [["RunStarted", "RunContent"]] and names(publisher)[-1] == "RunCompleted"
    assert publisher.outcome == "end"


def test_an_iterator_that_raises_after_the_held_ending_publishes_an_error_instead() -> None:
    publisher = RecordingPublisher()

    def events() -> Iterator[Any]:
        yield from ending_events()
        raise RuntimeError("after the ending")

    with pytest.raises(RuntimeError):
        list(publish(events(), publisher, before_close=lambda summary: None))
    assert names(publisher) == ["RunStarted", "RunContent", "RunError"] and publisher.outcome == "fail"


def test_a_failed_save_after_a_root_error_keeps_the_root_s_own_error() -> None:
    publisher = RecordingPublisher()
    events = [RunStartedEvent(run_id="r1"), RunErrorEvent(run_id="r1", content="model down", error_type="Provider")]

    def before_close(summary: RunSummary) -> None:
        raise OSError("database down")

    with pytest.raises(OSError):
        list(publish(iter(events), publisher, before_close=before_close))
    assert names(publisher) == ["RunStarted", "RunError"] and payloads(publisher)[-1]["error_type"] == "Provider"


@pytest.mark.parametrize(
    "options",
    [
        {"close_on_cancel": "End"},
        {"oversize": "split"},
        {"include": "RunContent"},
        {"protocol": "ag-ui", "transform": dict},
    ],
)
def test_a_refused_call_closes_the_stream_it_was_given(options: dict[str, Any]) -> None:
    publisher = RecordingPublisher()
    with pytest.raises(ValueError):
        publish(iter([]), publisher, **options)
    assert publisher.outcome == "fail" and publisher.close_calls == 1


async def test_a_refused_async_call_closes_its_stream_without_blocking_the_loop() -> None:
    publisher = AsyncRecordingPublisher()
    with pytest.raises(ValueError):
        apublish(iter_async([]), publisher, oversize="split")  # type: ignore[arg-type]
    given = AsyncRecordingPublisher()
    with pytest.raises(TypeError):
        publish(iter([]), given)  # type: ignore[arg-type]
    for _ in range(50):
        if publisher.outcome and given.outcome:
            break
        await asyncio.sleep(0.01)
    assert publisher.outcome == given.outcome == "fail"


def test_a_missing_ag_ui_extra_closes_the_stream_too(monkeypatch: pytest.MonkeyPatch) -> None:
    from iq_stream_agno import _tee

    def missing() -> None:
        raise ImportError("The AG-UI protocol needs the ag-ui extra")

    monkeypatch.setattr(_tee, "require_agui", missing)
    publisher = RecordingPublisher()
    with pytest.raises(ImportError):
        publish(iter([]), publisher, protocol="ag-ui")
    assert publisher.outcome == "fail"


async def iter_async(items: list[Any]) -> Any:
    for item in items:
        yield item
