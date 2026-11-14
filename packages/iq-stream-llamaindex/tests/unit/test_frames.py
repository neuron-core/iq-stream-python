"""Frames over the publisher's line limit: deltas are split, tool results cut, snapshots dropped, and every frame sent
fits. Run against a recorder with a 1,024-byte limit, the short server's."""

from __future__ import annotations

import json
import logging

import pytest
from checks import agui_events, assert_well_formed, joined
from fakes import BIG_RESULT, big_weather, get_weather, weather_agent
from iq_stream.testing import AsyncRecordingPublisher
from llama_index.protocols.ag_ui.agent import AGUIChatWorkflow
from workflows import Context, Workflow, step
from workflows.events import Event, StartEvent, StopEvent

from iq_stream_llamaindex import AGUIEncoder, Frame, NativeEncoder, publish
from iq_stream_llamaindex.encoders.base import bounded_json, cut, dumps, split_text

LIMIT = 1024
LONG_TOKEN = 'Paris is lovely in spring, «sunny» and "warm". ' * 120  # about 5.8 KB, quotes escaped twice


def fits_limit(publisher: AsyncRecordingPublisher) -> bool:
    return all(publisher.frame_size(frame.data, frame.event) <= LIMIT for frame in publisher.frames)


async def test_native_frames_split_long_deltas_and_cut_long_results_to_fit() -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=LIMIT)
    agent = weather_agent(tool=big_weather, answer_tokens=["Here: ", LONG_TOKEN])
    outcome = await publish(agent.run(user_msg="Weather?"), publisher)
    assert outcome.status == "completed" and publisher.outcome == "end" and fits_limit(publisher)
    streamed = [json.loads(frame.data) for frame in publisher.frames if frame.event == "AgentStream"]
    assert "".join(str(frame["delta"]) for frame in streamed).endswith("Here: " + LONG_TOKEN)
    (result,) = [json.loads(frame.data) for frame in publisher.frames if frame.event == "ToolCallResult"]
    assert result["truncated"] is True and str(result["content"]).endswith("…")
    assert BIG_RESULT.startswith(str(result["content"])[:-1]) and result["tool_id"] == "call_1"
    final = [json.loads(frame.data) for frame in publisher.frames if frame.event == "AgentOutput"][-1]
    assert final["truncated"] is True  # the final answer, repeated in AgentOutput, is cut


async def test_ag_ui_frames_split_long_deltas_and_arguments_and_cut_long_results_to_fit() -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=LIMIT)
    agent = weather_agent(tool=big_weather, answer_tokens=["Here: ", LONG_TOKEN], tool_kwargs={"city": "P" * 3000})
    await publish(agent.run(user_msg="Weather?"), publisher, encoder=AGUIEncoder(thread_id="t"))
    assert publisher.outcome == "end" and fits_limit(publisher)
    events = agui_events(publisher.data)
    assert_well_formed(events)
    assert joined(events, "TEXT_MESSAGE_CONTENT").endswith("Here: " + LONG_TOKEN)
    assert json.loads(joined(events, "TOOL_CALL_ARGS")) == {"city": "P" * 3000}
    assert sum(event["type"] == "TOOL_CALL_ARGS" for event in events) > 1
    (result,) = (event for event in events if event["type"] == "TOOL_CALL_RESULT")
    assert str(result["content"]).endswith("…") and result["toolCallId"] == "call_1"


async def test_a_snapshot_over_the_limit_is_dropped_with_a_warning_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    from ag_ui.core import RunAgentInput, UserMessage

    history = [UserMessage(id=f"u{i}", role="user", content="An earlier question. " * 10) for i in range(10)]
    workflow = AGUIChatWorkflow(llm=weather_agent().llm, backend_tools=[get_weather], timeout=30)
    run_input = RunAgentInput(
        thread_id="t", run_id="r", state={}, messages=history, tools=[], context=[], forwarded_props={}
    )
    publisher = AsyncRecordingPublisher(max_frame_bytes=LIMIT)
    with caplog.at_level(logging.WARNING, logger="iq_stream_llamaindex"):
        outcome = await publish(
            workflow.run(input_data=run_input), publisher, encoder=AGUIEncoder(thread_id="t", run_id="r")
        )
    assert outcome.status == "completed" and fits_limit(publisher)
    kinds = [event["type"] for event in agui_events(publisher.data)]
    assert "MESSAGES_SNAPSHOT" not in kinds and "TEXT_MESSAGE_CHUNK" in kinds and kinds[-1] == "RUN_FINISHED"
    assert "dropped a frame of MessagesSnapshotWorkflowEvent" in caplog.text
    assert "An earlier question" not in caplog.text


class Huge(Event):
    """An application event with nothing that can be cut: a long list of numbers."""


class Noisy(Workflow):
    @step
    async def report(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(Huge(numbers=list(range(2000))))
        ctx.write_event_to_stream(Huge(numbers=[1]))
        return StopEvent(result="done")


async def test_a_frame_that_cannot_be_made_to_fit_is_dropped_and_the_stream_continues(
    caplog: pytest.LogCaptureFixture,
) -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=LIMIT)
    with caplog.at_level(logging.WARNING, logger="iq_stream_llamaindex"):
        await publish(Noisy(timeout=10).run(), publisher)
    assert [frame.event for frame in publisher.frames] == ["RunStarted", "Huge", "StopEvent"]
    assert json.loads(publisher.frames[1].data) == {"numbers": [1]} and publisher.outcome == "end"
    assert "dropped a frame of Huge" in caplog.text


class Oversized(NativeEncoder):
    """Gives a fit() that does not fit: the frames must not be sent."""

    def fit(self, frame: Frame, fits: object) -> list[Frame]:
        return [frame]


async def test_a_fit_that_does_not_fit_is_not_trusted() -> None:
    publisher = AsyncRecordingPublisher(max_frame_bytes=LIMIT)
    await publish(weather_agent(tool=big_weather).run(user_msg="Weather?"), publisher, encoder=Oversized())
    assert "ToolCallResult" not in [frame.event for frame in publisher.frames] and fits_limit(publisher)


def test_cut_keeps_whole_characters() -> None:
    assert cut("aé€😀", 1) == "a"
    assert cut("aé€😀", 2) == "a"
    assert cut("aé€😀", 3) == "aé"
    assert cut("aé€😀", 6) == "aé€"
    assert cut("aé€😀", 100) == "aé€😀"


def test_split_text_gives_pieces_that_fit_and_rejoin() -> None:
    text = '«un» "deux" trois ' * 50

    def build(piece: str) -> Frame:
        return (None, dumps({"delta": piece}))

    pieces = split_text(text, build, lambda frame: len(frame[1].encode()) <= 64)
    assert pieces is not None and all(len(data.encode()) <= 64 for _, data in pieces)
    assert "".join(json.loads(data)["delta"] for _, data in pieces) == text
    assert split_text(text, build, lambda frame: False) is None


def test_bounded_json_cuts_strings_and_gives_up_on_what_strings_cannot_shrink() -> None:
    value, cut_any = bounded_json({"a": "x" * 100, "b": [1, 2]}, 40, "…")
    assert cut_any and value["b"] == [1, 2] and len(dumps(value).encode()) <= 40
    assert bounded_json({"a": "short"}, 40, "…") == ({"a": "short"}, False)
    assert bounded_json(list(range(100)), 40, "…") == ({}, True)
