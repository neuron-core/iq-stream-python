"""protocol="ag-ui": the frames of real CrewAI runs on fake models."""

from __future__ import annotations

import json
import typing
from datetime import datetime
from typing import Any

import ag_ui.core as ag
import pytest
from agui_checks import checked, conversation, parse, types
from crew_fakes import ANSWER, TERM, FakeReActLLM, FakeStreamingLLM, LookupTool, make_crew
from crewai import Agent, Crew, Task
from crewai.events import crewai_event_bus
from crewai.events.base_events import BaseEvent
from crewai.flow.flow import Flow, listen, start
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import publish


def run_crew(crew: Crew, **options: Any) -> tuple[RecordingPublisher, list[dict[str, Any]]]:
    publisher = RecordingPublisher()
    with publish(publisher, **options):
        crew.kickoff()
    return publisher, checked(publisher.data)


def test_one_tool_call_exact_sequence_and_end() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, thread_id="thread-1", run_id="run-1") as run:
        result = make_crew().kickoff()
    events = checked(publisher.data)
    assert types(events) == [
        "RUN_STARTED",
        "STEP_STARTED",
        "TOOL_CALL_START",
        *["TOOL_CALL_ARGS"] * 5,
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START",
        *["TEXT_MESSAGE_CONTENT"] * 6,
        "TEXT_MESSAGE_END",
        "STEP_FINISHED",
        "RUN_FINISHED",
    ]
    assert events[0] == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"}
    assert events[-1] == {"type": "RUN_FINISHED", "threadId": "thread-1", "runId": "run-1"}
    assert events[1] == {"type": "STEP_STARTED", "stepName": "task:Answer question demo"}
    talk = conversation(events)
    [tool_call_id] = talk.tool_args
    assert json.loads(talk.tool_args[tool_call_id]) == {"term": TERM}
    assert talk.tool_names[tool_call_id] == "lookup"
    assert talk.tool_results == {tool_call_id: f"value:{TERM}"}
    assert talk.text == ANSWER == result.raw
    assert list(talk.names.values()) == ["Researcher demo"]
    # The tool call belongs to the model call that streamed it; its result has a message id of its own.
    start_event = events[2]
    assert start_event["parentMessageId"] != events[10]["messageId"]
    assert events[9]["role"] == "tool"
    assert publisher.outcome == "end"
    assert run.result is not None and run.result.outcome == "end"
    assert run.frames_sent == len(events) == run.result.last_event_id
    assert run.errors == []


def test_frames_are_what_ag_ui_s_own_encoder_writes() -> None:
    models = {kind.model_fields["type"].default.value: kind for kind in typing.get_args(typing.get_args(ag.Event)[0])}
    publisher, events = run_crew(make_crew(thinking="Hmm."))
    for frame, event in zip(publisher.frames, events, strict=True):
        model = models[event["type"]].model_validate(event)
        assert frame.data == model.model_dump_json(by_alias=True, exclude_none=True)
        assert frame.event is None


def test_a_failing_model_ends_with_run_error_masked_and_fail() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError, match="fake provider outage") as raised, publish(publisher) as run:
        make_crew(fail_on_call=2).kickoff()
    events = checked(publisher.data)
    assert events[-1] == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "RuntimeError"}
    assert "sk-secret" not in "".join(publisher.data)
    assert publisher.outcome == "fail"
    assert type(raised.value) is RuntimeError
    assert run.errors == []


def test_expose_errors_publishes_the_message() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError), publish(publisher, expose_errors=True):
        make_crew(fail_on_call=1).kickoff()
    events = checked(publisher.data)
    assert events[-1] == {"type": "RUN_ERROR", "message": "fake provider outage at sk-secret", "code": "RuntimeError"}


def test_a_tool_that_raises_becomes_an_error_result_and_the_run_goes_on() -> None:
    publisher, events = run_crew(make_crew(tool=LookupTool(explode=True)))
    talk = conversation(events)
    [content] = talk.tool_results.values()
    assert json.loads(content) == {"error": "An error occurred."}
    assert "secret" not in "".join(publisher.data)
    # CrewAI hands the error to the model as the tool's observation, and the model answers.
    assert talk.text == ANSWER
    assert publisher.outcome == "end"


def test_parallel_tool_calls_get_their_own_results() -> None:
    terms = ["alpha", "beta", "gamma"]
    _, events = run_crew(make_crew(terms=terms, tool=LookupTool(seconds=0.05)))
    talk = conversation(events)
    assert len(talk.tool_args) == 3
    for tool_call_id, args in talk.tool_args.items():
        assert talk.tool_results[tool_call_id] == "value:" + json.loads(args)["term"]


def test_a_react_model_announces_its_tool_call_when_the_tool_starts() -> None:
    agent = Agent(role="Researcher", goal="Answer", backstory="A test agent", llm=FakeReActLLM(), tools=[LookupTool()])
    crew = Crew(agents=[agent], tasks=[Task(description="Answer", expected_output="One sentence", agent=agent)])
    _, events = run_crew(crew)
    talk = conversation(events)
    [(tool_call_id, args)] = talk.tool_args.items()
    assert json.loads(args) == {"term": "x"}
    assert talk.tool_results[tool_call_id] == "value:x"
    # The ReAct scratchpad is the model's text, published as it is.
    assert "Action: lookup" in talk.text and "Final Answer: It is value:x." in talk.text
    start_index = types(events).index("TOOL_CALL_START")
    assert types(events)[start_index : start_index + 4] == [
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
    ]


def test_a_model_without_streaming_publishes_each_answer_when_its_call_returns() -> None:
    _, events = run_crew(make_crew(stream=False))
    talk = conversation(events)
    assert talk.text == ANSWER
    assert types(events).count("TEXT_MESSAGE_CONTENT") == 1
    assert types(events).count("TOOL_CALL_START") == 1  # announced from tool_usage_started
    assert list(talk.tool_results.values()) == [f"value:{TERM}"]


def test_thinking_chunks_are_reasoning_closed_before_the_text() -> None:
    _, events = run_crew(make_crew(terms=[], thinking="Paris is the capital."))
    kinds = types(events)
    assert kinds[kinds.index("REASONING_START") : kinds.index("TEXT_MESSAGE_START")] == [
        "REASONING_START",
        "REASONING_MESSAGE_START",
        *["REASONING_MESSAGE_CONTENT"] * 4,
        "REASONING_MESSAGE_END",
        "REASONING_END",
    ]
    talk = conversation(events)
    assert list(talk.reasoning.values()) == ["Paris is the capital. "]
    assert talk.text == ANSWER


def test_a_flow_publishes_method_and_task_steps_and_not_the_agent_executor() -> None:
    class Research(Flow):  # type: ignore[type-arg]
        @start()
        def research(self) -> str:
            return make_crew().kickoff().raw

        @listen(research)
        def shout(self, text: str) -> str:
            return text.upper()

    publisher = RecordingPublisher()
    with publish(publisher):
        result = Research().kickoff()
    events = checked(publisher.data)
    assert result == ANSWER.upper()
    steps = [(event["type"], event["stepName"]) for event in events if event["type"].startswith("STEP_")]
    assert steps == [
        ("STEP_STARTED", "flow:research"),
        ("STEP_STARTED", "task:Answer question demo"),
        ("STEP_FINISHED", "task:Answer question demo"),
        ("STEP_FINISHED", "flow:research"),
        ("STEP_STARTED", "flow:shout"),
        ("STEP_FINISHED", "flow:shout"),
    ]
    native = RecordingPublisher()
    with publish(native, protocol="crewai"):
        Research().kickoff()
    flows = [json.loads(frame.data)["flow_name"] for frame in native.frames if frame.event == "flow_started"]
    assert flows == ["Research"]


def test_a_second_open_step_of_the_same_name_gets_a_suffix() -> None:
    publisher = RecordingPublisher()
    with publish(publisher):
        crews = [make_crew(name="same"), make_crew(name="same")]

        class Twice(Flow):  # type: ignore[type-arg]
            @start()
            def outer(self) -> str:
                # The second crew runs while the first one's task is open.
                crews[0].tasks[0].callback = lambda _: crews[1].kickoff()
                return crews[0].kickoff().raw

        Twice().kickoff()
    events = checked(publisher.data)
    names = [event["stepName"] for event in events if event["type"] == "STEP_STARTED"]
    assert names == ["flow:outer", "task:Answer question same", "task:Answer question same#2"]


def test_a_flow_paused_for_human_feedback_is_a_custom_event_and_ends() -> None:
    from crewai.flow.async_feedback import HumanFeedbackPending, PendingFeedbackContext
    from crewai.flow.human_feedback import human_feedback

    class Provider:
        def request_feedback(self, context: PendingFeedbackContext, flow: Flow[Any]) -> str:
            raise HumanFeedbackPending(context=context)

    class Review(Flow):  # type: ignore[type-arg]
        @start()
        @human_feedback(
            message="Approve the draft?",
            emit=["approved", "rejected"],
            llm=FakeStreamingLLM(answer="approved"),
            provider=Provider(),
        )
        def draft(self) -> str:
            return "a draft"

    publisher = RecordingPublisher()
    with publish(publisher) as run:
        outcome = Review().kickoff()
        saved = outcome  # the application persists the pending request before the close line
    assert isinstance(saved, HumanFeedbackPending)
    events = checked(publisher.data)
    [paused] = [event for event in events if event["type"] == "CUSTOM"]
    assert paused["name"] == "crewai.flow_paused"
    assert paused["value"]["message"] == "Approve the draft?"
    assert paused["value"]["emit"] == ["approved", "rejected"]
    assert paused["value"]["method_name"] == "draft"
    assert paused["value"]["flow_id"] == saved.context.flow_id
    assert events[-1]["type"] == "RUN_FINISHED"
    assert publisher.outcome == "end"
    assert run.errors == []


def test_a_feedback_request_is_a_custom_event() -> None:
    from crewai.events.types.flow_events import HumanFeedbackRequestedEvent

    publisher = RecordingPublisher()
    with publish(publisher):
        request = HumanFeedbackRequestedEvent(
            flow_name="Review", method_name="draft", output="a draft", message="Approve?", emit=["yes", "no"]
        )
        crewai_event_bus.emit(None, request)
    events = checked(publisher.data)
    assert events[1] == {
        "type": "CUSTOM",
        "name": "crewai.human_feedback_requested",
        "value": {"flow_name": "Review", "method_name": "draft", "message": "Approve?", "emit": ["yes", "no"]},
    }


class _Bridged(BaseEvent, ag.TextMessageChunkEvent):  # type: ignore[misc]
    """How flows written for ag-ui-crewai announce text: an event that is both CrewAI's and AG-UI's."""


def test_events_that_are_already_ag_ui_pass_through_unchanged() -> None:
    publisher = RecordingPublisher()
    with publish(publisher):
        crewai_event_bus.emit(None, _Bridged(type=ag.EventType.TEXT_MESSAGE_CHUNK, message_id="m1", delta="Hi"))
    events = parse(publisher.data)
    assert events[1] == {"type": "TEXT_MESSAGE_CHUNK", "messageId": "m1", "delta": "Hi"}


def test_include_and_exclude_choose_the_events_the_encoder_sees() -> None:
    # Without the chunks, each answer is published when its call returns, as for a model without streaming.
    _, events = run_crew(make_crew(), exclude=["llm_stream_chunk"])
    assert types(events).count("TEXT_MESSAGE_CONTENT") == 1
    assert conversation(events).text == ANSWER
    _, events = run_crew(make_crew(), include=["task_started", "task_completed"])
    assert types(events) == ["RUN_STARTED", "STEP_STARTED", "STEP_FINISHED", "RUN_FINISHED"]
    with pytest.raises(ValueError, match="Unknown CrewAI event type"):
        publish(RecordingPublisher(), include=["llm_stream_chunks"])


def test_a_model_that_reuses_tool_call_ids_gets_unique_ones() -> None:
    llm = FakeStreamingLLM(tool_args=[{"term": "a"}], answer="done", fixed_ids=True)
    crew = make_crew(llm=llm)
    publisher = RecordingPublisher()
    with publish(publisher):
        crew.kickoff()
        llm.calls = 0  # the second kickoff streams the same tool call id again
        crew.kickoff()
    events = checked(publisher.data)
    starts = [event["toolCallId"] for event in events if event["type"] == "TOOL_CALL_START"]
    assert len(starts) == 2 and len(set(starts)) == 2


def _tool_chunk(call_id: str, tool_call_id: str, chunk: str) -> Any:
    from crewai.events.types.llm_events import LLMStreamChunkEvent

    function = {"name": "lookup", "arguments": chunk}
    return LLMStreamChunkEvent(
        chunk=chunk, call_id=call_id, tool_call={"id": tool_call_id, "function": function, "index": 0}
    )


def test_a_failed_call_s_tool_call_never_gets_a_later_result() -> None:
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallFailedEvent, LLMCallType
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent

    args = {"term": "x"}
    publisher = RecordingPublisher()
    with publish(publisher):
        emit = crewai_event_bus.emit
        emit(None, _tool_chunk("first", "tc-first", json.dumps(args)))
        emit(None, LLMCallFailedEvent(error="boom", call_id="first"))
        emit(None, _tool_chunk("retry", "tc-retry", json.dumps(args)))
        emit(None, LLMCallCompletedEvent(response=[], call_type=LLMCallType.TOOL_CALL, call_id="retry"))
        emit(None, ToolUsageStartedEvent(tool_name="lookup", tool_args=args))
        now = datetime.now()
        emit(
            None,
            ToolUsageFinishedEvent(tool_name="lookup", tool_args=args, output="ok", started_at=now, finished_at=now),
        )
    talk = conversation(checked(publisher.data))
    assert talk.tool_results == {"tc-retry": "ok"}


def test_a_result_without_a_started_tool_is_announced_with_it() -> None:
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent

    publisher = RecordingPublisher()
    with publish(publisher):
        now = datetime.now()
        finished = ToolUsageFinishedEvent(
            tool_name="lookup", tool_args={"term": "y"}, output="v", started_at=now, finished_at=now
        )
        crewai_event_bus.emit(None, finished)
    events = checked(publisher.data)
    assert types(events) == [
        "RUN_STARTED",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "RUN_FINISHED",
    ]
    talk = conversation(events)
    assert list(talk.tool_args.values()) == ['{"term":"y"}'] and list(talk.tool_results.values()) == ["v"]


def test_a_tool_s_events_bind_to_the_call_of_their_own_agent() -> None:
    from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallStartedEvent, LLMCallType
    from crewai.events.types.tool_usage_events import ToolUsageFinishedEvent, ToolUsageStartedEvent

    args = {"term": "x"}
    now = datetime.now()
    publisher = RecordingPublisher()
    with publish(publisher):
        emit = crewai_event_bus.emit
        for agent in ("A", "B"):  # two agents stream a call of the same tool with the same arguments
            call_id = f"call-{agent}"
            emit(None, LLMCallStartedEvent(call_id=call_id, agent_id=agent, messages=[]))
            emit(None, _tool_chunk(call_id, f"tc-{agent}", json.dumps(args)))
            emit(None, LLMCallCompletedEvent(response=[], call_type=LLMCallType.TOOL_CALL, call_id=call_id))
        for agent in ("B", "A"):  # B's tool runs first
            emit(None, ToolUsageStartedEvent(tool_name="lookup", tool_args=args, agent_id=agent))
            finished = ToolUsageFinishedEvent(
                tool_name="lookup",
                tool_args=args,
                agent_id=agent,
                output=f"by {agent}",
                started_at=now,
                finished_at=now,
            )
            emit(None, finished)
    talk = conversation(checked(publisher.data))
    assert talk.tool_results == {"tc-A": "by A", "tc-B": "by B"}
