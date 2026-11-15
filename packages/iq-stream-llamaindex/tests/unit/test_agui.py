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
    direct_agent,
    get_weather,
    handoff_workflow,
    parallel_agent,
    repeated_id_agent,
    tool_call,
    weather_agent,
)
from iq_stream.testing import AsyncRecordingPublisher
from llama_index.core.agent.workflow import AgentStream, FunctionAgent, ToolCall
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


async def test_a_return_direct_tools_result_is_the_assistants_answer() -> None:
    events = await run(direct_agent().run(user_msg="Weather in Paris?"))
    assert_well_formed(events)
    kinds = types(events)
    assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + TOOL_RESULT
    assert kinds.index("TOOL_CALL_RESULT") < kinds.index("TEXT_MESSAGE_START", kinds.index("TOOL_CALL_RESULT"))
    assert kinds[-2:] == ["STEP_FINISHED", "RUN_FINISHED"]


@pytest.mark.parametrize("parallel", [False, True], ids=["two-steps", "one-step"])
async def test_tool_calls_that_share_an_id_are_each_announced_under_an_id_of_their_own(parallel: bool) -> None:
    events = await run(repeated_id_agent(parallel=parallel).run(user_msg="Paris and Rome?"))
    assert_well_formed(events)
    starts = [event["toolCallId"] for event in events if event["type"] == "TOOL_CALL_START"]
    assert sorted(starts) == ["get_weather", "get_weather#2"]  # type: ignore[type-var]
    cities = {e["toolCallId"]: json.loads(str(e["delta"]))["city"] for e in events if e["type"] == "TOOL_CALL_ARGS"}
    results = {event["toolCallId"]: event["content"] for event in events if event["type"] == "TOOL_CALL_RESULT"}
    assert sorted(cities.values()) == ["Paris", "Rome"]
    assert results == {call: f"21 °C and sunny in {city}" for call, city in cities.items()}  # each its own result


async def test_one_encoder_serves_runs_one_after_another() -> None:
    encoder = AGUIEncoder(thread_id="thread-1")
    for n in (1, 2):
        publisher = AsyncRecordingPublisher()
        await publish(
            weather_agent().run(user_msg="Weather?", run_id=f"run-{n}-000000000000"), publisher, encoder=encoder
        )
        events = agui_events(publisher.data)
        assert_well_formed(events)
        assert events[0]["runId"] == events[-1]["runId"] == f"run-{n}-000000000000"
        assert [event["toolCallId"] for event in events if event["type"] == "TOOL_CALL_START"] == ["call_1"]
    fixed = AGUIEncoder(thread_id="thread-1", run_id="chosen")
    for _ in (1, 2):
        assert json.loads(next(iter(fixed.start("handler-run")))[1])["runId"] == "chosen"


async def test_a_lone_surrogate_in_the_answer_becomes_a_replacement_character_and_the_run_stays_well_formed() -> None:
    agent = weather_agent(answer_tokens=["Sunny \ud83d", "\ude00 in ", "Paris."])
    events = await run(agent.run(user_msg="Weather?"))
    assert_well_formed(events)
    assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT + "Sunny �� in Paris."


async def test_a_lone_surrogate_in_a_tool_result_does_not_lose_the_result() -> None:
    async def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "21 \udcc3C in Paris"  # decoded with surrogateescape

    events = await run(weather_agent(tool=get_weather).run(user_msg="Weather?"))
    (result,) = (event for event in events if event["type"] == "TOOL_CALL_RESULT")
    assert result["content"] == "21 �C in Paris"


class Flaky(AGUIEncoder):
    """Fails once, on the first text delta, after it has built its frames."""

    failed = False

    def _stream(self, event: AgentStream) -> list[tuple[str | None, str]]:
        frames = super()._stream(event)
        if event.delta and not self.failed:
            self.failed = True
            raise ValueError("cannot encode this")
        return frames


async def test_an_event_that_fails_to_encode_leaves_no_trace_in_the_encoders_state() -> None:
    publisher = AsyncRecordingPublisher()
    await publish(weather_agent().run(user_msg="Weather?"), publisher, encoder=Flaky(thread_id="t"))
    events = agui_events(publisher.data)
    assert_well_formed(events)  # the next delta opens the message the failed one did not
    assert joined(events, "TEXT_MESSAGE_CONTENT") == FIRST_TEXT[len("Let ") :] + ANSWER


async def test_a_tool_call_of_a_step_without_text_has_no_parent_message() -> None:
    llm = ScriptedLLM(
        script=[
            [TextBlock(text="Let me check."), tool_call("get_weather", "c1", city="Paris")],
            [tool_call("get_weather", "c2", city="Rome")],
            [TextBlock(text="Both sunny.")],
        ]
    )
    events = await run(FunctionAgent(name="assistant", tools=[get_weather], llm=llm).run(user_msg="Paris, Rome?"))
    first, second = (event for event in events if event["type"] == "TOOL_CALL_START")
    assert first["parentMessageId"] == next(e["messageId"] for e in events if e["type"] == "TEXT_MESSAGE_START")
    assert "parentMessageId" not in second


def test_non_finite_numbers_are_null_in_custom_events() -> None:
    encoder = AGUIEncoder(thread_id="t")
    (frame,) = encoder.encode(AgentStreamStructuredOutput(output={"confidence": float("nan"), "max": float("inf")}))
    assert json.loads(frame[1], parse_constant=_refuse)["value"] == {"confidence": None, "max": None}
    (start, arguments, _) = encoder.encode(ToolCall(tool_name="t", tool_kwargs={"x": float("-inf")}, tool_id="c"))
    assert json.loads(json.loads(arguments[1])["delta"], parse_constant=_refuse) == {"x": None}
    assert json.loads(start[1])["toolCallName"] == "t"


def _refuse(constant: str) -> None:
    raise AssertionError(f"{constant} is not JSON")
