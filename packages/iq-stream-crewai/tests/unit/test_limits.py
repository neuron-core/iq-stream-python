"""Frames over the publisher's max_frame_bytes: deltas split, values cut with a marker, the rest dropped."""

from __future__ import annotations

import json
import time

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import FakeStreamingLLM, LookupTool, make_crew
from crewai import Agent, Crew, Task
from crewai.events.types.llm_events import LLMStreamChunkEvent
from iq_stream import FrameTooLarge
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import publish
from iq_stream_crewai._agui import AgUiEncoder
from iq_stream_crewai._limits import Fitter, dumps
from iq_stream_crewai._snapshot import EVENT_FIELDS, Snapshotter

# Text that costs the most per character on the wire: escapes, multi-byte and astral characters.
AWKWARD = 'a"\\\n\x01é€😀/'


def sizes_fit(publisher: RecordingPublisher) -> None:
    for frame in publisher.frames:
        assert publisher.frame_size(frame.data, frame.event) <= publisher.max_frame_bytes


@pytest.mark.parametrize("limit", [1024, 65_536])
def test_a_200_kb_tool_result_and_a_100_kb_delta(limit: int) -> None:
    answer = ("word " + AWKWARD) * (100_000 // (5 + len(AWKWARD)))
    llm = FakeStreamingLLM(tool_args=[{"term": "big"}], answer=answer)
    # One delta of 100 KB: the model streams the whole answer as one word.
    llm.answer = answer.replace(" ", "_")
    publisher = RecordingPublisher(max_frame_bytes=limit)
    with publish(publisher, max_text_chars=None) as run:
        make_crew(llm=llm, tool=LookupTool(result="r" * 200_000)).kickoff()
    sizes_fit(publisher)
    events = checked(publisher.data)
    talk = conversation(events)
    assert talk.text == llm.answer  # the pieces of the delta concatenate to the original
    assert types(events).count("TEXT_MESSAGE_CONTENT") > 1
    [result] = talk.tool_results.values()
    assert result.startswith("r" * 100) and result.endswith("characters truncated]")
    assert run.errors == []


def test_max_text_chars_cuts_values_before_the_frame_limit() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, max_text_chars=100):
        make_crew(tool=LookupTool(result="r" * 200_000)).kickoff()
    [result] = conversation(checked(publisher.data)).tool_results.values()
    assert result == "r" * 100 + "… [199900 characters truncated]"


def test_native_chunks_are_split_and_outputs_cut() -> None:
    llm = FakeStreamingLLM(tool_args=[{"term": "big"}], answer=AWKWARD * 500)
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publish(publisher, protocol="crewai", max_text_chars=None) as run:
        make_crew(llm=llm, tool=LookupTool(result="r" * 5000)).kickoff()
    sizes_fit(publisher)
    chunks = [json.loads(frame.data) for frame in publisher.frames if frame.event == "llm_stream_chunk"]
    chunks = [chunk for chunk in chunks if "tool_call" not in chunk]
    assert "".join(chunk["chunk"] for chunk in chunks) == AWKWARD * 500
    [finished] = [json.loads(f.data) for f in publisher.frames if f.event == "tool_usage_finished"]
    assert finished["output"].endswith("characters truncated]")
    assert run.errors == []


def test_long_labels_and_error_messages_fit() -> None:
    llm = FakeStreamingLLM(tool_args=[], fail_on_call=1)
    agent = Agent(role="R" * 3000, goal="g", backstory="b", llm=llm, tools=[], max_retry_limit=0)
    crew = Crew(agents=[agent], tasks=[Task(description="D" * 3000, expected_output="o", agent=agent)])
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with pytest.raises(RuntimeError), publish(publisher, expose_errors=True, max_text_chars=None) as run:
        crew.kickoff()
    sizes_fit(publisher)
    events = checked(publisher.data)
    step = next(event["stepName"] for event in events if event["type"] == "STEP_STARTED")
    assert step.startswith("task:DDD") and len(step) == 120
    assert next(event["name"] for event in events if event["type"] == "TEXT_MESSAGE_START") == "R" * 119 + "…"
    assert events[-1]["type"] == "RUN_ERROR"
    assert run.errors == []


def test_the_fitter_splits_and_cuts_awkward_text() -> None:
    dropped: list[FrameTooLarge] = []
    from iq_stream._frame import frame_size

    fitter = Fitter(300, frame_size, dropped.append)
    text = AWKWARD * 200
    pieces = fitter.split({"type": "TEXT_MESSAGE_CONTENT", "messageId": "m", "delta": text}, "delta")
    assert len(pieces) > 1
    assert all(frame_size(data, event) <= 300 for event, data in pieces)
    assert "".join(json.loads(data)["delta"] for _, data in pieces) == text
    [(_, data)] = fitter.cut({"type": "X", "content": text, "other": {"k": text}}, ["content", "other"])
    assert frame_size(data) <= 300
    value = json.loads(data)
    assert value["content"].endswith("characters truncated]") and value["other"].endswith("characters truncated]")
    # A frame that cannot fit even with everything cut is dropped and reported.
    assert fitter.whole({"type": "X", "id": "i" * 400}) == []
    assert fitter.split({"type": "X", "id": "i" * 400, "delta": "abc"}, "delta") == []
    assert [error.limit for error in dropped] == [300, 300]
    assert dumps({"é": "/"}) == '{"é":"/"}'


def test_the_cost_per_token_stays_small() -> None:
    # The sink runs in the model's thread; the plan measured 7 µs per token for the snapshot and the AG-UI encoding.
    snapshot = Snapshotter(frozenset(EVENT_FIELDS), 16_000, False)
    encoder = AgUiEncoder(snapshot, Fitter(65_536, lambda d, e: len(d), lambda e: None), "t", "r", False)
    event = LLMStreamChunkEvent(chunk="hello ", call_id="c1", agent_id="a", task_id="t")
    event.emission_sequence = 1
    count = 5000
    started = time.perf_counter()
    for _ in range(count):
        encoder.frames(encoder.snapshot(None, event))  # type: ignore[arg-type]
    per_token = (time.perf_counter() - started) / count
    assert per_token < 200e-6, f"{per_token * 1e6:.1f} µs per token"


def test_the_fitter_measures_every_frame_that_might_not_fit() -> None:
    # Frames short enough are sent without measuring; a character of the data can take up to six bytes in the line
    # (a lone surrogate, escaped), three for most CJK text and four for an emoji.
    from iq_stream._frame import frame_size

    for limit in (1024, 4096):
        fitter = Fitter(limit, frame_size, lambda error: None)
        for char in ("é", "中", "😀", "\ud800", '"', "\\"):
            for count in range(50, 1600, 25):
                payload = {"type": "X", "delta": char * count}
                for frames in (fitter.whole(payload), fitter.split(payload, "delta"), fitter.cut(payload, ["delta"])):
                    assert all(frame_size(data, event) <= limit for event, data in frames), (limit, char, count)
