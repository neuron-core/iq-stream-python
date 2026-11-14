"""AGUIEncoder: every frame is an AG-UI event the ag_ui.core models accept, in an order a client can apply."""

from __future__ import annotations

import json
import sys

import pytest
from ag_ui.core import RunAgentInput, UserMessage
from checks import agui_events, assert_well_formed, joined, types
from fakes import (
    ANSWER,
    FIRST_TEXT,
    THINKING,
    TOOL_RESULT,
    ScriptedLLM,
    broken_weather,
    get_weather,
    handoff_workflow,
    parallel_agent,
    weather_agent,
)
from iq_stream.testing import AsyncRecordingPublisher
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.agent.workflow.workflow_events import AgentStreamStructuredOutput
from llama_index.core.base.llms.types import TextBlock
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from iq_stream_llamaindex import AGUIEncoder, publish

Events = list[dict[str, object]]


async def run(handler: object, **options: object) -> Events:
    publisher = AsyncRecordingPublisher()
    encoder = AGUIEncoder(thread_id="thread-1", **options)  # type: ignore[arg-type]
    await publish(handler, publisher, encoder=encoder)  # type: ignore[arg-type]
    assert all(frame.event is None for frame in publisher.frames)  # unnamed, read in onmessage
    return agui_events(publisher.data)


async def test_an_agent_run_is_a_well_formed_ag_ui_run() -> None:
    events = await run(weather_agent().run(user_msg="Weather in Paris?", run_id="run-0000000000000001"))
    assert_well_formed(events)
    assert events[0] == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-0000000000000001"}
    assert events[-1] == {"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-0000000000000001"}
    assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + ANSWER
    start = next(event for event in events if event["type"] == "TOOL_CALL_START")
    assert start["toolCallName"] == "get_weather" and start["toolCallId"] == "call_1"
    assert json.loads(joined(events, "TOOL_CALL_ARGS")) == {"city": "Paris"}
    (result,) = (event for event in events if event["type"] == "TOOL_CALL_RESULT")
    assert result["toolCallId"] == "call_1" and result["content"] == TOOL_RESULT and result["role"] == "tool"


async def test_a_tool_call_belongs_to_the_assistant_message_of_its_model_step() -> None:
    events = await run(weather_agent().run(user_msg="Weather?"))
    first_text = next(event for event in events if event["type"] == "TEXT_MESSAGE_START")
    start = next(event for event in events if event["type"] == "TOOL_CALL_START")
    assert start["parentMessageId"] == first_text["messageId"]


async def test_reasoning_opens_and_closes_before_the_text_message() -> None:
    events = await run(weather_agent().run(user_msg="Weather?"))
    kinds = types(events)
    reasoning = kinds[kinds.index("REASONING_START") : kinds.index("REASONING_END") + 1]
    assert reasoning == [
        "REASONING_START",
        "REASONING_MESSAGE_START",
        "REASONING_MESSAGE_CONTENT",
        "REASONING_MESSAGE_END",
        "REASONING_END",
    ]
    assert kinds.index("REASONING_END") < kinds.index("TEXT_MESSAGE_START")
    assert joined(events, "REASONING_MESSAGE_CONTENT") == THINKING
    started = next(event for event in events if event["type"] == "REASONING_MESSAGE_START")
    assert started["role"] == "reasoning"


async def test_each_agent_of_an_agent_workflow_is_a_step_and_the_hand_off_a_tool_call() -> None:
    events = await run(handoff_workflow().run(user_msg="Weather in Rome?"))
    assert_well_formed(events)
    steps = [(event["type"], event["stepName"]) for event in events if "stepName" in event]
    assert steps == [
        ("STEP_STARTED", "triage"),
        ("STEP_FINISHED", "triage"),
        ("STEP_STARTED", "weather"),
        ("STEP_FINISHED", "weather"),
    ]
    calls = [event["toolCallName"] for event in events if event["type"] == "TOOL_CALL_START"]
    assert calls == ["handoff", "get_weather"]
    kinds = types(events)
    handoff_result = next(i for i, event in enumerate(events) if event.get("toolCallId") == "h1" and "content" in event)
    assert handoff_result < kinds.index("STEP_FINISHED")
    assert joined(events, "TEXT_MESSAGE_CONTENT") == "Sunny in Rome."


async def test_steps_can_be_left_out() -> None:
    events = await run(handoff_workflow().run(user_msg="Weather in Rome?"), emit_steps=False)
    assert not {"STEP_STARTED", "STEP_FINISHED"} & set(types(events))
    assert_well_formed(events)


async def test_parallel_tool_calls_interleave_by_completion_and_stay_well_formed() -> None:
    events = await run(parallel_agent().run(user_msg="Rome and Oslo?"))
    assert_well_formed(events)
    order = [(event["type"], event["toolCallId"]) for event in events if event["type"] == "TOOL_CALL_RESULT"]
    assert order == [("TOOL_CALL_RESULT", "call_b"), ("TOOL_CALL_RESULT", "call_a")]


async def test_an_agent_that_does_not_stream_still_publishes_its_answer_once() -> None:
    llm = ScriptedLLM(script=[[TextBlock(text="A plain answer.")]])
    events = await run(FunctionAgent(name="plain", tools=[], llm=llm, streaming=False).run(user_msg="Hi"))
    assert_well_formed(events)
    assert types(events).count("TEXT_MESSAGE_START") == 1
    assert joined(events, "TEXT_MESSAGE_CONTENT") == "A plain answer."


async def test_a_tool_error_is_masked_unless_errors_are_exposed() -> None:
    masked = await run(weather_agent(tool=broken_weather).run(user_msg="Weather?"))
    (result,) = (event for event in masked if event["type"] == "TOOL_CALL_RESULT")
    assert result["content"] == "An error occurred."
    exposed = await run(weather_agent(tool=broken_weather).run(user_msg="Weather?"), expose_errors=True)
    (result,) = (event for event in exposed if event["type"] == "TOOL_CALL_RESULT")
    assert "weather service down" in str(result["content"])


async def test_a_failed_run_exposes_the_models_error_only_when_asked() -> None:
    events = await run(weather_agent(fail_on_second_call=True).run(user_msg="Weather?"), expose_errors=True)
    assert events[-1]["type"] == "RUN_ERROR" and "exploded" in str(events[-1]["message"])
    assert_well_formed(events)


async def test_ag_ui_events_of_ag_ui_chat_workflow_pass_through_unchanged() -> None:
    workflow = AGUIChatWorkflow(llm=weather_agent().llm, backend_tools=[get_weather], timeout=30)
    run_input = RunAgentInput(
        thread_id="thread-1",
        run_id="run-1",
        state={},
        messages=[UserMessage(id="user-1", role="user", content="Weather in Paris?")],
        tools=[],
        context=[],
        forwarded_props={},
    )
    handler = workflow.run(input_data=run_input)
    publisher = AsyncRecordingPublisher()
    seen: list[str] = []

    def collect(event: Event) -> None:
        if hasattr(event, "model_dump_json") and not isinstance(event, StopEvent):
            seen.append(event.model_dump_json(by_alias=True))

    await publish(handler, publisher, encoder=AGUIEncoder(thread_id="thread-1", run_id="run-1"), on_event=collect)
    events = agui_events(publisher.data)
    assert events[0]["type"] == "RUN_STARTED" and events[-1]["type"] == "RUN_FINISHED"
    assert publisher.data[1:-1] == seen  # byte for byte
    assert {"TEXT_MESSAGE_CHUNK", "TOOL_CALL_CHUNK", "MESSAGES_SNAPSHOT", "STATE_SNAPSHOT"} <= set(types(events))


class Progress(Event):
    step: int


class Report(Workflow):
    @step
    async def run_report(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(Progress(step=1, note="halfway"))
        return StopEvent(result="done")


async def test_an_applications_own_event_is_a_custom_event_or_nothing() -> None:
    events = await run(Report(timeout=10).run())
    assert types(events) == ["RUN_STARTED", "CUSTOM", "RUN_FINISHED"]
    assert events[1] == {"type": "CUSTOM", "name": "Progress", "value": {"step": 1, "note": "halfway"}}
    events = await run(Report(timeout=10).run(), custom_events=False)
    assert types(events) == ["RUN_STARTED", "RUN_FINISHED"]


def test_a_structured_output_is_a_custom_event() -> None:
    encoder = AGUIEncoder(thread_id="t")
    frames = list(encoder.encode(AgentStreamStructuredOutput(output={"city": "Paris", "celsius": 21})))
    assert [json.loads(data) for _, data in frames] == [
        {"type": "CUSTOM", "name": "structured_output", "value": {"city": "Paris", "celsius": 21}}
    ]


def test_without_ag_ui_protocol_the_encoder_says_which_extra_to_install(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "ag_ui.core", None)
    with pytest.raises(ImportError, match=r"iq-stream-llamaindex\[agui\]"):
        AGUIEncoder(thread_id="t")


def test_max_field_bytes_is_positive() -> None:
    with pytest.raises(ValueError, match="max_field_bytes"):
        AGUIEncoder(thread_id="t", max_field_bytes=0)
