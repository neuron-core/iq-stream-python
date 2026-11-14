"""transport="bus": handlers on CrewAI's global event bus, with a slow neighbouring listener that reorders them."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Iterator

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, TERM, LookupTool, make_crew
from crewai.events import crewai_event_bus
from crewai.events.base_events import BaseEvent
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import publish
from iq_stream_crewai._snapshot import EVENT_FIELDS
from iq_stream_crewai._transport import _event_classes


@pytest.fixture(autouse=True)
def slow_neighbour() -> Iterator[None]:
    """A listener that takes 5 ms on every event but stream chunks, which the bus runs inline."""

    def neighbour(source: object, event: BaseEvent) -> None:
        time.sleep(0.005)

    kinds = [
        kind
        for kind in set(_event_classes())
        if getattr(kind.model_fields.get("type"), "default", None) in set(EVENT_FIELDS) - {"llm_stream_chunk"}
    ]
    for kind in kinds:
        crewai_event_bus.register_handler(kind, neighbour)
    yield
    for kind in kinds:
        crewai_event_bus.off(kind, neighbour)


def test_one_tool_call_in_order() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, transport="bus") as run:
        result = make_crew(delay=0.002).kickoff()
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
    talk = conversation(events)
    assert talk.text == ANSWER == result.raw
    assert list(talk.tool_results.values()) == [f"value:{TERM}"]
    assert publisher.outcome == "end" and run.errors == []


def test_native_frames_in_emission_order() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai", transport="bus"):
        make_crew(delay=0.002).kickoff()
    seqs = [json.loads(data)["seq"] for data in publisher.data]
    assert seqs == sorted(seqs)
    assert publisher.frames[-1].event == "crew_kickoff_completed"


def test_a_failing_model() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError), publish(publisher, transport="bus"):
        make_crew(fail_on_call=2).kickoff()
    events = checked(publisher.data)
    assert events[-1] == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "RuntimeError"}
    assert publisher.outcome == "fail"


def test_parallel_tool_calls() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, transport="bus"):
        make_crew(terms=["alpha", "beta", "gamma"], tool=LookupTool(seconds=0.02)).kickoff()
    talk = conversation(checked(publisher.data))
    assert len(talk.tool_args) == 3
    for tool_call_id, args in talk.tool_args.items():
        assert talk.tool_results[tool_call_id] == "value:" + json.loads(args)["term"]


def test_two_runs_in_two_threads() -> None:
    answers = ["ALPHA alpha alpha.", "BETA beta beta."]
    publishers = [RecordingPublisher(), RecordingPublisher()]

    def work(index: int) -> None:
        with publish(publishers[index], transport="bus"):
            make_crew(name=str(index), answer=answers[index], delay=0.002).kickoff()

    threads = [threading.Thread(target=work, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    for publisher, answer in zip(publishers, answers, strict=True):
        assert conversation(checked(publisher.data)).text == answer
