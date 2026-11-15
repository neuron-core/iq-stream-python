"""The tee with AG-UI events from Agno's own converter: the sequence, the terminal event, the size policy, and every
frame valid against the ag_ui.core models."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import pytest
from ag_ui.core import Event
from agno.agent import Agent
from agno.os.interfaces.agui.stream import stream_agno_response_as_agui_events
from agno.workflow import Workflow
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from pydantic import TypeAdapter
from recorded import Errors, collect, payloads, types
from scripted_model import SLOW, team, weather_agent

from iq_stream_agno import MASKED_ERROR, apublish, publish

EVENTS: TypeAdapter[Any] = TypeAdapter(Event)
QUIET = ("ModelRequestStarted", "ModelRequestCompleted")


def valid(publisher: RecordingPublisher | AsyncRecordingPublisher) -> list[Any]:
    """Every frame parsed by the ag_ui.core models: what a Python AG-UI client accepts."""
    return [EVENTS.validate_json(data) for data in publisher.data]


def text_of(content: str) -> str:
    """A tool result as Agno's converter writes it: agno 3 JSON-encodes a string result, agno 2.9 does not."""
    try:
        decoded = json.loads(content)
    except ValueError:
        return content
    return decoded if isinstance(decoded, str) else content


def collapse(sequence: list[str]) -> list[str]:
    """Runs of text events as one "text", for the plan's sequence."""
    text = {"TEXT_MESSAGE_START", "TEXT_MESSAGE_CONTENT", "TEXT_MESSAGE_END"}
    out: list[str] = []
    for kind in sequence:
        label = "text" if kind in text else kind
        if not out or out[-1] != label or label != "text":
            out.append(label)
    return out


def test_the_weather_run_has_the_plan_s_sequence() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent()
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, protocol="agui", exclude=QUIET))
    assert collapse(types(publisher, raw=False)) == [
        "RUN_STARTED",
        "text",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "text",
        "RUN_FINISHED",
    ]
    events = valid(publisher)
    started, finished = payloads(publisher)[0], payloads(publisher)[-1]
    assert started["runId"] == finished["runId"] == events[1].event["run_id"]
    args = next(payload for payload in payloads(publisher) if payload["type"] == "TOOL_CALL_ARGS")
    assert json.loads(args["delta"]) == {"city": "Paris"}
    result = next(payload for payload in payloads(publisher) if payload["type"] == "TOOL_CALL_RESULT")
    assert text_of(result["content"]) == "21C and sunny in Paris"
    assert publisher.outcome == "end"


def test_the_same_events_through_agno_s_own_stream_give_the_same_sequence() -> None:
    events = list(weather_agent().run("Weather?", stream=True, stream_events=True))
    publisher = RecordingPublisher()
    list(publish(iter(events), publisher, protocol="agui"))
    agno_types = [
        str(event.type.value)
        for event in stream_agno_response_as_agui_events(iter(events), thread_id="t", run_id=events[0].run_id)
    ]
    assert types(publisher)[0] == "RUN_STARTED" and types(publisher)[1:] == agno_types


async def test_a_failed_run_ends_with_a_masked_run_error() -> None:
    publisher = AsyncRecordingPublisher()
    stream = weather_agent(fail_on_turn=2).arun("Weather?", stream=True, stream_events=True)
    await collect(apublish(stream, publisher, protocol="agui"))
    valid(publisher)
    last = payloads(publisher)[-1]
    assert last["type"] == "RUN_ERROR" and last["message"] == MASKED_ERROR and last["code"] == "RuntimeError"
    assert "scripted model failure" not in "".join(publisher.data)
    assert publisher.outcome == "fail"


async def test_a_cancelled_run_finishes_with_the_cancelled_outcome() -> None:
    SLOW["seconds"] = 0.3
    publisher = AsyncRecordingPublisher()
    stream = weather_agent(tool_name="slow_weather").arun(
        "Weather?", stream=True, stream_events=True, run_id="agui-cancel"
    )
    async for event in apublish(stream, publisher, protocol="agui"):
        if event.event == "ToolCallStarted":
            Agent.cancel_run("agui-cancel")
    valid(publisher)
    last = payloads(publisher)[-1]
    assert last["type"] == "RUN_FINISHED" and last["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "fail"


def test_an_iterator_that_raises_ends_with_run_error() -> None:
    publisher = RecordingPublisher()

    def events() -> Any:
        for index, event in enumerate(weather_agent().run("Weather?", stream=True, stream_events=True)):
            if index == 4:
                raise RuntimeError("iterator blew up")
            yield event

    with pytest.raises(RuntimeError):
        list(publish(events(), publisher, protocol="agui"))
    valid(publisher)
    last = payloads(publisher)[-1]
    assert last["type"] == "RUN_ERROR" and last["message"] == MASKED_ERROR and publisher.outcome == "fail"


def test_a_caller_that_stops_finishes_the_run_as_cancelled_with_its_spans_closed() -> None:
    publisher = RecordingPublisher()
    stream = publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher, protocol="agui")
    for event in stream:
        if event.event == "RunContent":
            break
    stream.close()
    valid(publisher)
    assert types(publisher, raw=False)[-2:] == ["TEXT_MESSAGE_END", "RUN_FINISHED"]
    assert payloads(publisher)[-1]["outcome"] == {"type": "cancelled"} and publisher.outcome == "fail"


def test_oversize_arguments_are_split_and_an_oversize_result_is_cut() -> None:
    city = "Paris " + "x" * 3_000
    publisher = RecordingPublisher(max_frame_bytes=1_024)
    errors = Errors()
    agent = weather_agent(tool_args={"city": city})
    stream = agent.run("Weather?", stream=True, stream_events=True)
    list(publish(stream, publisher, protocol="agui", exclude=QUIET, on_error=errors))
    valid(publisher)
    assert all(publisher.frame_size(data) <= 1_024 for data in publisher.data)
    args = [payload["delta"] for payload in payloads(publisher) if payload["type"] == "TOOL_CALL_ARGS"]
    assert len(args) > 1 and json.loads("".join(args)) == {"city": city}
    (result,) = [payload["content"] for payload in payloads(publisher) if payload["type"] == "TOOL_CALL_RESULT"]
    assert result.lstrip('"').startswith("21C and sunny in Paris") and result.endswith(" [truncated]")
    assert types(publisher)[-1] == "RUN_FINISHED" and publisher.outcome == "end"
    # What does not fit and cannot be split (Agno's RAW copies of the tool events) is dropped and reported.
    assert all(error.code is None and "limit is 1024" in str(error) for error in errors)


def test_event_names_name_each_frame_after_its_type() -> None:
    publisher = RecordingPublisher()
    list(publish(weather_agent().run("Weather?", stream=True), publisher, protocol="agui", event_names=True))
    assert [frame.event for frame in publisher.frames] == types(publisher)


def test_a_team_run_converts_and_ends_with_the_leader() -> None:
    publisher = RecordingPublisher()
    list(publish(team().run("Weather?", stream=True, stream_events=True), publisher, protocol="agui"))
    valid(publisher)
    kinds = types(publisher, raw=False)
    assert kinds[0] == "RUN_STARTED" and kinds[-1] == "RUN_FINISHED" and kinds.count("RUN_FINISHED") == 1
    assert "TOOL_CALL_RESULT" in kinds and publisher.outcome == "end"


async def test_a_team_whose_member_speaks_first_ends_with_the_leader_s_error_under_the_team_s_run_id() -> None:
    publisher = AsyncRecordingPublisher()
    received = await collect(
        apublish(team(leader_fails=True, preamble=False).arun("Weather?", stream=True), publisher, protocol="agui")
    )
    valid(publisher)
    team_run_id = received[-1].run_id
    assert received[-1].event == "TeamRunError" and received[0].run_id != team_run_id
    rows = payloads(publisher)
    assert rows[0] == {"type": "RUN_STARTED", "threadId": received[0].session_id, "runId": team_run_id}
    # The member's RunCompleted is not the run's end: it goes out as RAW, and the leader's error ends the run.
    assert [row["event"]["event"] for row in rows if row["type"] == "RAW"].count("RunCompleted") == 1
    assert "RUN_FINISHED" not in types(publisher) and rows[-1]["type"] == "RUN_ERROR"
    assert publisher.outcome == "fail"


def test_a_failing_tool_s_error_text_is_masked_in_every_frame() -> None:
    publisher = RecordingPublisher()
    agent = weather_agent(tool_name="broken_tool")
    list(publish(agent.run("Weather?", stream=True, stream_events=True), publisher, protocol="agui"))
    valid(publisher)
    assert "secret database password" not in "".join(publisher.data)
    (result,) = [row for row in payloads(publisher) if row["type"] == "TOOL_CALL_RESULT"]
    assert text_of(result["content"]) == MASKED_ERROR
    (error,) = [
        row["event"] for row in payloads(publisher) if row["type"] == "RAW" and row["event"]["event"] == "ToolCallError"
    ]
    assert error["error"] == MASKED_ERROR and error["tool"]["result"] == MASKED_ERROR
    assert publisher.outcome == "end"


async def test_a_workflow_s_error_text_is_masked_in_ag_ui_too() -> None:
    async def steps(workflow: Workflow, execution_input: Any) -> AsyncIterator[str]:
        raise RuntimeError("workflow blew up")
        yield ""

    publisher = AsyncRecordingPublisher()
    workflow = Workflow(id="failing-wf", name="Failing WF", steps=steps, telemetry=False)
    with pytest.raises(RuntimeError, match="workflow blew up"):
        await collect(apublish(workflow.arun("Go", stream=True, stream_events=True), publisher, protocol="agui"))
    valid(publisher)
    assert "workflow blew up" not in "".join(publisher.data)
    assert payloads(publisher)[-1]["type"] == "RUN_ERROR" and publisher.outcome == "fail"
