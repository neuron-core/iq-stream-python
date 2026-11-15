"""Order under concurrency: parallel tool calls, two runs of one agent at once, and a generator advanced from
several threads."""

from __future__ import annotations

import asyncio
import threading

from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from recorded import collect, names, payloads
from scripted_model import SLOW, weather_agent

from iq_stream_agno import apublish, publish


async def test_parallel_tool_calls_arrive_started_started_completed_completed() -> None:
    SLOW["seconds"] = 0.2
    publisher = AsyncRecordingPublisher()
    agent = weather_agent(tool_name="slow_weather", n_tool_calls=2)
    received = await collect(apublish(agent.arun("Weather?", stream=True, stream_events=True), publisher))
    tools = [(row["event"], row["tool"]["tool_call_id"]) for row in payloads(publisher) if "ToolCall" in row["event"]]
    assert tools == [
        ("ToolCallStarted", "call_1"),
        ("ToolCallStarted", "call_2"),
        ("ToolCallCompleted", "call_1"),
        ("ToolCallCompleted", "call_2"),
    ]
    assert names(publisher) == [event.event for event in received]


async def test_two_concurrent_runs_of_one_agent_do_not_mix() -> None:
    SLOW["seconds"] = 0.2
    agent = weather_agent(tool_name="slow_weather", n_tool_calls=2)
    publishers = [AsyncRecordingPublisher(), AsyncRecordingPublisher()]

    async def one(index: int) -> list[str]:
        stream = agent.arun("Weather?", stream=True, stream_events=True, run_id=f"run-{index}")
        return [event.event async for event in apublish(stream, publishers[index])]

    seen = await asyncio.gather(one(0), one(1))
    for index, publisher in enumerate(publishers):
        assert {row["run_id"] for row in payloads(publisher)} == {f"run-{index}"}
        assert names(publisher) == seen[index] and publisher.outcome == "end"


def test_a_generator_advanced_from_several_threads_keeps_the_order() -> None:
    publisher = RecordingPublisher()
    stream = publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher)
    received: list[str] = []
    threads: set[str] = set()
    done = threading.Event()

    def step() -> None:
        threads.add(threading.current_thread().name)  # idents are reused, names are not
        try:
            received.append(next(stream).event)
        except StopIteration:
            done.set()

    while not done.is_set():  # each event on a thread of its own, as a thread pool may do
        thread = threading.Thread(target=step)
        thread.start()
        thread.join(10)
    assert len(threads) > 1
    assert names(publisher) == received and publisher.outcome == "end"
