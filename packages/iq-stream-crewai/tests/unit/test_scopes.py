"""What CrewAI does for itself is not the application's run: the flows it runs inside agents and memory, and the
model calls of memory's analysis, are left out; the application's own model calls are published."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, FakeMemoryLLM, FakeStreamingLLM, embed, make_crew
from crewai.flow.flow import Flow, start
from crewai.memory.recall_flow import RecallFlow
from crewai.memory.unified_memory import Memory
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import EVENT_TYPES, publish
from iq_stream_crewai._limits import Fitter
from iq_stream_crewai._native import NativeEncoder
from iq_stream_crewai._snapshot import Snapshotter

Transport = Literal["sink", "bus"]


@pytest.fixture
def memory(tmp_path: Path) -> Iterator[Memory]:
    memory = Memory(llm=FakeMemoryLLM(), embedder=embed, storage=str(tmp_path / "memory"))
    yield memory
    memory.close()


@pytest.mark.parametrize("transport", ["sink", "bus"])
def test_a_crew_s_memory_publishes_neither_its_model_calls_nor_its_flows(memory: Memory, transport: Transport) -> None:
    crew = make_crew()
    crew.memory = memory
    publisher = RecordingPublisher()
    with publish(publisher, transport=transport) as run:
        crew.kickoff()
    memory_llm = memory._llm  # pyright: ignore[reportPrivateUsage]
    assert isinstance(memory_llm, FakeMemoryLLM) and memory_llm.calls >= 2  # extraction and analysis ran
    events = checked(publisher.data)
    talk = conversation(events)
    assert list(talk.texts.values()) == [ANSWER]  # the agent's answer, and no memory JSON after it
    assert types(events).count("TEXT_MESSAGE_START") == 1
    assert run.errors == []

    crew = make_crew()
    crew.memory = memory
    native = RecordingPublisher()
    with publish(native, protocol="crewai", transport=transport):
        crew.kickoff()
    names = [frame.event for frame in native.frames]
    assert not [name for name in names if name and name.startswith("flow_")]  # RecallFlow, EncodingFlow
    models = {json.loads(frame.data)["model"] for frame in native.frames if frame.event == "llm_call_started"}
    assert models == {"fake-stream"}
    assert names.count("llm_call_completed") == 2


def test_an_internal_flow_s_model_calls_are_left_out_even_when_they_stream() -> None:
    analysis = json.dumps({"keywords": ["paris"], "suggested_scopes": ["/"], "complexity": "simple"})

    class Storage:
        def search(self, *args: object, **kwargs: object) -> list[object]:
            return []

        def list_scopes(self, *args: object, **kwargs: object) -> list[str]:
            return ["/"]

        def get_scope_info(self, *args: object, **kwargs: object) -> None:
            return None

    llm = FakeStreamingLLM(answer=analysis)
    publisher = RecordingPublisher()
    with publish(publisher):
        # A query over the analysis threshold: the flow asks its model.
        RecallFlow(storage=Storage(), llm=llm, embedder=embed).kickoff(inputs={"query": "What about Paris? " * 20})
    assert llm.calls == 1
    assert types(checked(publisher.data)) == ["RUN_STARTED", "RUN_FINISHED"]


def test_the_application_s_own_model_calls_are_published() -> None:
    llm = FakeStreamingLLM(answer="Said directly.")

    class Direct(Flow):  # type: ignore[type-arg]
        @start()
        def ask(self) -> str:
            return llm.call("Say it")

    publisher = RecordingPublisher()
    with publish(publisher):
        llm.call("Say it")
        Direct().kickoff()
    events = checked(publisher.data)
    assert list(conversation(events).texts.values()) == ["Said directly.", "Said directly."]


def test_scopes_are_forgotten_when_they_close() -> None:
    snapshotter = Snapshotter(frozenset(EVENT_TYPES), None, False)
    publisher = RecordingPublisher()
    encoder = NativeEncoder(snapshotter, Fitter(publisher.max_frame_bytes, publisher.frame_size, lambda error: None))
    with publish(publisher, protocol=encoder):
        make_crew().kickoff()
        assert snapshotter._machinery == set()  # pyright: ignore[reportPrivateUsage]
    assert publisher.frames[-1].event == "crew_kickoff_completed"


def test_a_feedback_classification_in_a_flow_method_is_published_as_the_readme_says() -> None:
    from crewai.flow.human_feedback import human_feedback

    class Provider:
        def request_feedback(self, context: object, flow: object) -> str:
            return "Looks good to me"

    class Review(Flow):  # type: ignore[type-arg]
        @start()
        @human_feedback(
            message="Approve?",
            emit=["approved", "rejected"],
            llm=FakeStreamingLLM(answer="approved"),
            provider=Provider(),
        )
        def draft(self) -> str:
            return "a draft"

    publisher = RecordingPublisher()
    with publish(publisher):
        Review().kickoff()
    # CrewAI's call, inside the application's method: nothing tells it from a call of the application's.
    assert list(conversation(checked(publisher.data)).texts.values()) == ["approved"]
