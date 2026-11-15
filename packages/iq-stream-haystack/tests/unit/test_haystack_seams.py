"""How IQStream meets Haystack: callback selection, the pipeline's deep copy, serialization, contexts, threads,
sub-agents and async runs."""

from __future__ import annotations

import asyncio
import copy
import inspect
import threading
import time
from collections.abc import Hashable
from typing import Any

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall, select_streaming_callback
from haystack.tools import tool
from haystack.utils import deserialize_callable, serialize_callable
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import (
    Frame,
    IQStream,
    current_stream,
    iq_streaming_callback,
    iq_streaming_callback_async,
)
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import (
    ScriptedChatGenerator,
    SyncOnlyChatGenerator,
    text_turn,
    tool_turn,
    weather_call,
    weather_script,
)

try:
    from haystack.tools import AgentTool
except ImportError:  # haystack-ai releases before AgentTool
    AgentTool = None

needs_agent_tool = pytest.mark.skipif(AgentTool is None, reason="this haystack-ai has no AgentTool")

QUESTION = [ChatMessage.from_user("Weather in Paris and Rome?")]


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny in {city}"


def two_city_script() -> list[Any]:
    calls = [weather_call("Paris", "call_p"), weather_call("Rome", "call_r")]
    return [
        tool_turn(calls, text=["Let ", "me ", "check."], reasoning=["The user ", "wants weather."]),
        text_turn(["Both ", "sunny."]),
    ]


def test_callback_and_acallback_are_what_haystack_selects_for_sync_and_async_runs() -> None:
    stream = IQStream(RecordingPublisher())
    assert select_streaming_callback(None, stream.callback, requires_async=False) == stream.callback
    assert select_streaming_callback(None, stream.acallback, requires_async=True) == stream.acallback
    with pytest.raises(ValueError, match="coroutine"):
        select_streaming_callback(None, stream.acallback, requires_async=False)


def test_a_deep_copy_of_the_callback_still_publishes_to_the_stream() -> None:
    stream = IQStream(RecordingPublisher())
    assert copy.deepcopy(stream.callback).__self__ is stream  # pyright: ignore[reportFunctionMemberAccess]
    assert copy.copy(stream) is stream


def test_a_pipeline_run_with_the_callback_in_its_data_publishes_every_chunk() -> None:
    pipe = Pipeline()
    pipe.add_component("agent", Agent(chat_generator=ScriptedChatGenerator(weather_script()), tools=[get_weather]))
    publisher = RecordingPublisher()
    with IQStream(publisher) as stream:
        result = pipe.run({"agent": {"messages": QUESTION, "streaming_callback": stream.callback}})
    conversation = verify_agui(publisher.data)
    assert conversation.texts == ["Let me check.", "It is sunny in Paris."]
    assert result["agent"]["last_message"].text == "It is sunny in Paris."


def test_the_module_level_callback_serializes_and_loads_from_yaml_without_an_allowlist() -> None:
    path = "haystack_integrations.iq_stream.callbacks.iq_streaming_callback"
    assert serialize_callable(iq_streaming_callback) == path
    assert deserialize_callable(path) is iq_streaming_callback

    # Everything in this pipeline loads from Haystack's default allowlist: the callback's namespace is on it.
    pipe = Pipeline()
    generator = MockChatGenerator(responses=["It is sunny in Paris."])
    pipe.add_component("agent", Agent(chat_generator=generator, streaming_callback=iq_streaming_callback))
    yaml = pipe.dumps()
    assert f"streaming_callback: {path}" in yaml
    loaded = Pipeline.loads(yaml)

    publisher = RecordingPublisher()
    with IQStream(publisher):  # no callback at run time: the component's own finds the stream
        loaded.run({"agent": {"messages": QUESTION}})
    assert verify_agui(publisher.data).texts == ["It is sunny in Paris."]


def test_the_module_level_callback_does_nothing_outside_a_stream() -> None:
    assert current_stream() is None
    iq_streaming_callback(StreamingChunk(content="nobody listens"))
    asyncio.run(iq_streaming_callback_async(StreamingChunk(content="nobody listens")))


def test_runs_in_two_threads_publish_to_their_own_streams_through_the_module_level_callback() -> None:
    publishers = [RecordingPublisher(), RecordingPublisher()]
    barrier = threading.Barrier(2)

    def run(publisher: RecordingPublisher, city: str) -> None:
        agent = Agent(
            chat_generator=ScriptedChatGenerator(weather_script(city=city), delay=0.005),
            tools=[get_weather],
            streaming_callback=iq_streaming_callback,
        )
        with IQStream(publisher) as stream:
            barrier.wait(5)
            assert current_stream() is stream
            agent.run(QUESTION)

    threads = [
        threading.Thread(target=run, args=(p, city)) for p, city in zip(publishers, ["Paris", "Rome"], strict=True)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(10)
    assert [verify_agui(p.data).texts[-1] for p in publishers] == ["It is sunny in Paris.", "It is sunny in Rome."]


async def test_runs_in_two_tasks_publish_to_their_own_streams_through_the_module_level_callback() -> None:
    async def run(city: str) -> RecordingPublisher:
        agent = Agent(
            chat_generator=ScriptedChatGenerator(weather_script(city=city), delay=0.005),
            tools=[get_weather],
            streaming_callback=iq_streaming_callback_async,
        )
        publisher = RecordingPublisher()
        async with IQStream(publisher):
            await agent.run_async(QUESTION)
        return publisher

    paris, rome = await asyncio.gather(run("Paris"), run("Rome"))
    assert verify_agui(paris.data).texts[-1] == "It is sunny in Paris."
    assert verify_agui(rome.data).texts[-1] == "It is sunny in Rome."


def test_eight_threads_calling_back_at_once_lose_and_tear_no_frame() -> None:
    publisher = RecordingPublisher()
    words = [f"w{n} " for n in range(40)]
    with IQStream(publisher, protocol="ai-sdk") as stream:
        barrier = threading.Barrier(8)

        def produce() -> None:
            barrier.wait(5)
            for chunk in text_turn(words).chunks:
                stream.callback(chunk)
                time.sleep(0.001)  # so that the eight messages really are open at once

        threads = [threading.Thread(target=produce) for _ in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
    message = verify_ai_sdk(publisher.data)
    assert message.texts == ["".join(words)] * 8
    assert publisher.outcome == "end"


def test_the_converter_is_called_one_chunk_at_a_time_from_concurrent_callers() -> None:
    class Watching:
        """A converter that sees whether two calls ever overlap."""

        def __init__(self) -> None:
            self.inside = 0
            self.most = 0

        def start(self) -> list[Frame]:
            return []

        def convert(self, chunk: StreamingChunk, source: Hashable) -> list[Frame]:
            self.inside += 1
            self.most = max(self.most, self.inside)
            time.sleep(0.002)
            self.inside -= 1
            return [Frame(chunk.content)]

        def finish(self) -> list[Frame]:
            return []

        error = cancelled = custom = finish  # pyright: ignore[reportAssignmentType]

    watching = Watching()
    publisher = RecordingPublisher()
    with IQStream(publisher, protocol=watching) as stream:
        barrier = threading.Barrier(4)

        def produce(thread: int) -> None:
            barrier.wait(5)
            for n in range(10):
                stream.callback(StreamingChunk(content=f"{thread}:{n}"))

        threads = [threading.Thread(target=produce, args=(thread,)) for thread in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
    assert watching.most == 1
    for thread in range(4):  # each caller's frames in the order it called back
        assert [d for d in publisher.data if d.startswith(f"{thread}:")] == [f"{thread}:{n}" for n in range(10)]


@needs_agent_tool
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_parallel_sub_agents_whose_messages_interleave_stay_apart(mode: str) -> None:
    @tool
    def slow_weather(city: str) -> str:
        """Get the weather for a city, slowly."""
        time.sleep(0.05)
        return f"Sunny in {city}"

    slow_weather.name = "get_weather"

    def research(n: int) -> Any:
        # Text, then a tool call, then the answer, a few milliseconds apart: the two sub-agents overlap.
        call = weather_call(f"City{n}", f"weather_{n}")
        script = [tool_turn([call], text=[f"s{n}a ", f"s{n}b ", f"s{n}c"]), text_turn([f"Sub{n} ", "is ", "done."])]
        sub = Agent(chat_generator=ScriptedChatGenerator(script, delay=0.003), tools=[slow_weather])
        assert AgentTool is not None
        return AgentTool(agent=sub, name=f"research{n}", description="Research a question")

    calls = [
        ToolCall(tool_name=f"research{n}", arguments={"messages": [{"role": "user", "content": "q"}]}, id=f"r{n}")
        for n in range(2)
    ]
    coordinator = Agent(
        chat_generator=ScriptedChatGenerator([tool_turn(calls, shape="mock"), text_turn(["All ", "done."])]),
        tools=[research(0), research(1)],
        tool_streaming_callback_passthrough=True,
    )
    publisher = RecordingPublisher()
    if mode == "sync":  # the sub-agents call back from two tool threads
        with IQStream(publisher) as stream:
            await asyncio.to_thread(coordinator.run, [ChatMessage.from_user("go")], streaming_callback=stream.callback)
    else:  # from two tasks in one thread
        async with IQStream(publisher) as stream:
            await coordinator.run_async([ChatMessage.from_user("go")], streaming_callback=stream.acallback)
    conversation = verify_agui(publisher.data)
    assert sorted(conversation.texts) == ["All done.", "Sub0 is done.", "Sub1 is done.", "s0a s0b s0c", "s1a s1b s1c"]
    assert conversation.results == {
        "weather_0": "Sunny in City0",
        "weather_1": "Sunny in City1",
        "r0": "Sub0 is done.",
        "r1": "Sub1 is done.",
    }


@needs_agent_tool
@pytest.mark.parametrize("mode", ["sync", "async"])
async def test_parallel_sub_agents_with_passthrough_publish_a_valid_flat_stream(mode: str) -> None:
    def sub_reply(messages: list[ChatMessage]) -> ChatMessage:
        if any(message.tool_call_result for message in messages):
            return ChatMessage.from_assistant("Sub-agent: it is sunny.")
        return ChatMessage.from_assistant(tool_calls=[ToolCall(tool_name="get_weather", arguments={"city": "Paris"})])

    sub = Agent(chat_generator=MockChatGenerator(response_fn=sub_reply), tools=[get_weather])
    assert AgentTool is not None
    research = AgentTool(agent=sub, name="research", description="Research a question")
    calls = [
        ToolCall(
            tool_name="research", arguments={"messages": [{"role": "user", "content": f"q{n}"}]}, id=f"research_{n}"
        )
        for n in range(2)
    ]
    coordinator = Agent(
        chat_generator=ScriptedChatGenerator([tool_turn(calls, shape="mock"), text_turn(["Coordinator: ", "done."])]),
        tools=[research],
        tool_streaming_callback_passthrough=True,
    )
    publisher = RecordingPublisher()
    if mode == "sync":
        with IQStream(publisher) as stream:
            await asyncio.to_thread(coordinator.run, [ChatMessage.from_user("go")], streaming_callback=stream.callback)
    else:
        async with IQStream(publisher) as stream:
            await coordinator.run_async([ChatMessage.from_user("go")], streaming_callback=stream.acallback)
    conversation = verify_agui(publisher.data)
    assert len(conversation.calls) == 4  # two research calls, and one weather call in each sub-agent
    assert len(conversation.results) == 4
    assert conversation.texts.count("Sub-agent: it is sunny.") == 2
    assert conversation.texts[-1] == "Coordinator: done."


async def test_an_async_run_with_reasoning_and_parallel_tools() -> None:
    agent = Agent(chat_generator=ScriptedChatGenerator(two_city_script(), delay=0.001), tools=[get_weather])
    publisher = RecordingPublisher()
    async with IQStream(publisher) as stream:
        await agent.run_async(QUESTION, streaming_callback=stream.acallback)
    conversation = verify_agui(publisher.data)
    assert conversation.reasoning == ["The user wants weather."]
    assert conversation.results == {"call_p": "Sunny in Paris", "call_r": "Sunny in Rome"}
    assert publisher.outcome == "end"


async def test_a_sync_only_generator_in_an_async_run_publishes_through_the_sync_callback() -> None:
    agent = Agent(chat_generator=SyncOnlyChatGenerator(two_city_script()), tools=[get_weather])
    publisher = RecordingPublisher()
    async with IQStream(publisher) as stream:
        with pytest.raises(ValueError, match="coroutine"):  # Haystack's own rule, which the README states
            await agent.run_async(QUESTION, streaming_callback=stream.acallback)
    publisher = RecordingPublisher()
    async with IQStream(publisher) as stream:
        await agent.run_async(QUESTION, streaming_callback=stream.callback)
    assert verify_agui(publisher.data).results == {"call_p": "Sunny in Paris", "call_r": "Sunny in Rome"}


def test_tee_publishes_then_calls_the_application_s_own_callback() -> None:
    publisher = RecordingPublisher()
    seen: list[tuple[int, str]] = []
    with IQStream(publisher) as stream:
        both = stream.tee(lambda chunk: seen.append((publisher.frames_sent, chunk.content)))
        Agent(chat_generator=ScriptedChatGenerator(weather_script())).run(QUESTION, streaming_callback=both)
    contents = [content for _, content in seen]
    assert "".join(contents).startswith("Let me check.")
    assert seen[0][0] >= 2  # RUN_STARTED, then the chunk's own frames, before the application saw it


async def test_tee_of_an_async_callback_is_async() -> None:
    publisher = RecordingPublisher()
    seen: list[str] = []

    async def mine(chunk: StreamingChunk) -> None:
        seen.append(chunk.content)

    async with IQStream(publisher) as stream:
        both = stream.tee(mine)
        assert inspect.iscoroutinefunction(both)
        await Agent(chat_generator=ScriptedChatGenerator(weather_script())).run_async(QUESTION, streaming_callback=both)
    assert "".join(seen).startswith("Let me check.")
    assert verify_agui(publisher.data).texts[0] == "Let me check."
