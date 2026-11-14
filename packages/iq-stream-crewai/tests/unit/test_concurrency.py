"""Runs side by side in one process, and events emitted by several threads into one stream."""

from __future__ import annotations

import asyncio
import contextvars
import json
import threading

from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, LookupTool, make_crew
from crewai import Crew
from crewai.events import crewai_event_bus
from crewai.events.types.llm_events import LLMCallCompletedEvent, LLMCallType, LLMStreamChunkEvent
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import publish

ANSWERS = ["ALPHA alpha alpha alpha alpha.", "BETA beta beta beta beta."]


def assert_isolated(publishers: list[RecordingPublisher]) -> None:
    for publisher, answer in zip(publishers, ANSWERS, strict=True):
        talk = conversation(checked(publisher.data))
        assert talk.text == answer
        assert list(talk.tool_results.values()) == [f"value:{answer.split()[0]}"]
        assert publisher.outcome == "end"


def crew(index: int) -> Crew:
    term = ANSWERS[index].split()[0]
    return make_crew(name=str(index), answer=ANSWERS[index], terms=[term], tool=LookupTool(seconds=0.05), delay=0.002)


def test_two_runs_in_two_threads() -> None:
    publishers = [RecordingPublisher(), RecordingPublisher()]

    def work(index: int) -> None:
        with publish(publishers[index]):
            crew(index).kickoff()

    threads = [threading.Thread(target=work, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    assert_isolated(publishers)


async def test_two_akickoffs_in_one_loop() -> None:
    publishers = [RecordingPublisher(), RecordingPublisher()]

    async def one(index: int) -> None:
        async with publish(publishers[index]):
            await crew(index).akickoff()

    await asyncio.wait_for(asyncio.gather(one(0), one(1)), 30)
    assert_isolated(publishers)


async def test_two_kickoff_asyncs_in_one_loop() -> None:
    publishers = [RecordingPublisher(), RecordingPublisher()]

    async def one(index: int) -> None:
        async with publish(publishers[index]):
            await crew(index).kickoff_async()

    await asyncio.wait_for(asyncio.gather(one(0), one(1)), 30)
    assert_isolated(publishers)


def test_crew_stream_true_iterated_inside_the_block() -> None:
    publisher = RecordingPublisher()
    with publish(publisher):
        streaming = make_crew(crew_stream=True, tool=LookupTool(seconds=0.02)).kickoff()
        own = "".join(chunk.content for chunk in streaming if chunk.chunk_type.value == "text")
        result = streaming.result
    events = checked(publisher.data)
    assert own == ANSWER == result.raw
    talk = conversation(events)
    assert talk.text == ANSWER
    assert "TOOL_CALL_RESULT" in types(events)


def test_events_from_many_threads_keep_each_thread_s_order_and_none_is_lost() -> None:
    threads, chunks = 8, 200
    publisher = RecordingPublisher()
    barrier = threading.Barrier(threads)

    def emit(index: int) -> None:
        barrier.wait(5)
        call_id = f"call-{index}"
        for number in range(chunks):
            crewai_event_bus.emit(None, LLMStreamChunkEvent(chunk=f"{index}:{number};", call_id=call_id))
        completed = LLMCallCompletedEvent(response="", call_type=LLMCallType.LLM_CALL, call_id=call_id)
        crewai_event_bus.emit(None, completed)

    with publish(publisher):
        workers = [
            threading.Thread(target=contextvars.copy_context().run, args=(emit, index)) for index in range(threads)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(30)
    events = checked(publisher.data)
    talk = conversation(events)
    for index in range(threads):
        assert talk.texts[f"call-{index}"] == "".join(f"{index}:{number};" for number in range(chunks))
    assert len(events) == 2 + threads * (chunks + 2)


def test_parallel_tool_results_keep_each_event_s_frames_together() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai"):
        make_crew(terms=["a", "b", "c", "d"], tool=LookupTool(seconds=0.02)).kickoff()
    names = [frame.event for frame in publisher.frames]
    data = [json.loads(frame.data) for frame in publisher.frames]
    starts = [item["tool_args"]["term"] for name, item in zip(names, data, strict=True) if name == "tool_usage_started"]
    finishes = [item["output"] for name, item in zip(names, data, strict=True) if name == "tool_usage_finished"]
    assert sorted(starts) == ["a", "b", "c", "d"]
    assert sorted(finishes) == ["value:a", "value:b", "value:c", "value:d"]
    assert names.count("crew_kickoff_completed") == 1 and names[-1] == "crew_kickoff_completed"
