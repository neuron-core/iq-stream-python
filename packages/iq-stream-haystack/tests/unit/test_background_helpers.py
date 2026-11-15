"""start_stream and start_stream_sync, with the client's publisher replaced by a recorder."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Any

import pytest
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import FallbackChatGenerator
from haystack.dataclasses import ChatMessage, StreamingChunk, ToolCall
from haystack.tools import tool
from iq_stream import IQClient
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import hayhooks
from haystack_integrations.iq_stream.hayhooks import start_stream, start_stream_sync
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import ScriptedChatGenerator, SyncOnlyChatGenerator, Turn, text_turn, tool_turn, weather_script

try:
    from haystack.tools import AgentTool
except ImportError:  # haystack-ai releases before AgentTool
    AgentTool = None

QUESTION = [ChatMessage.from_user("Weather in Paris?")]
SECRET = "test-app-secret-0123456789abcdef0123"


@pytest.fixture
def recorders(monkeypatch: pytest.MonkeyPatch) -> dict[str, RecordingPublisher]:
    """The publishers the client would have opened, by stream id."""
    opened: dict[str, RecordingPublisher] = {}

    def publisher(self: IQClient, stream_id: str) -> RecordingPublisher:
        opened[stream_id] = RecordingPublisher(stream_id)
        return opened[stream_id]

    monkeypatch.setattr(IQClient, "publisher", publisher)
    return opened


@pytest.fixture
def client() -> IQClient:
    return IQClient("http://127.0.0.1:9", "demo-app", SECRET, public_url="https://stream.example.com")


def slow_agent(release: threading.Event, generator: Any = None) -> Agent:
    @tool
    def get_weather(city: str) -> str:
        """Waits for the test."""
        assert release.wait(5)
        return f"Sunny in {city}"

    return Agent(chat_generator=generator or ScriptedChatGenerator(weather_script()), tools=[get_weather])


async def wait_for(condition: Any, seconds: float = 5) -> None:
    deadline = time.monotonic() + seconds
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


async def finished(recorders: dict[str, RecordingPublisher], stream_id: str) -> None:
    """Waits until the stream is closed and the task that ran it has ended."""
    await wait_for(lambda: stream_id in recorders and recorders[stream_id].outcome is not None)
    await wait_for(lambda: not hayhooks._TASKS)  # pyright: ignore[reportPrivateUsage]


async def test_start_stream_answers_at_once_saves_before_the_close_and_keeps_the_task(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    saved: list[tuple[str, object]] = []

    async def before_close(stream_id: str, result: dict[str, Any]) -> None:
        saved.append((stream_id, recorders[stream_id].outcome))

    ticket = await start_stream(
        client, slow_agent(release), {"messages": QUESTION}, before_close=before_close, run_id="r1"
    )
    stream_id = str(ticket["streamId"])
    assert ticket["url"] == f"https://stream.example.com/{stream_id}?token={ticket['token']}"
    assert client.verify("sub", str(ticket["token"]), stream_id)
    assert len(hayhooks._TASKS) == 1  # pyright: ignore[reportPrivateUsage]
    await wait_for(lambda: stream_id in recorders and recorders[stream_id].frames_sent > 5)
    assert recorders[stream_id].outcome is None and saved == []
    release.set()
    await wait_for(lambda: recorders[stream_id].outcome is not None)
    assert saved == [(stream_id, None)]
    assert recorders[stream_id].outcome == "end"
    assert verify_agui(recorders[stream_id].data).terminal["runId"] == "r1"
    await wait_for(lambda: not hayhooks._TASKS)  # pyright: ignore[reportPrivateUsage]


async def test_a_failing_run_closes_with_fail_and_is_logged(
    client: IQClient, recorders: dict[str, RecordingPublisher], caplog: pytest.LogCaptureFixture
) -> None:
    script = [Turn(text_turn(["x"]).chunks, ChatMessage.from_assistant("x"), error=RuntimeError("boom"), fail_after=0)]
    agent = Agent(chat_generator=ScriptedChatGenerator(script))
    with caplog.at_level(logging.ERROR, "haystack_integrations.iq_stream"):
        ticket = await start_stream(client, agent, {"messages": QUESTION})
        stream_id = str(ticket["streamId"])
        await finished(recorders, stream_id)
        await wait_for(lambda: not hayhooks._TASKS)  # pyright: ignore[reportPrivateUsage]
    assert recorders[stream_id].outcome == "fail"
    assert any("the run failed" in record.getMessage() for record in caplog.records)


async def test_a_save_that_fails_closes_with_fail(client: IQClient, recorders: dict[str, RecordingPublisher]) -> None:
    def before_close(stream_id: str, result: dict[str, Any]) -> None:
        raise OSError("database down")

    release = threading.Event()
    release.set()
    ticket = await start_stream(client, slow_agent(release), {"messages": QUESTION}, before_close=before_close)
    stream_id = str(ticket["streamId"])
    await finished(recorders, stream_id)
    assert recorders[stream_id].outcome == "fail"


async def test_a_pipeline_gets_the_callback_in_every_streaming_component(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    release.set()
    pipe = Pipeline()
    pipe.add_component("agent", slow_agent(release))
    mine: list[StreamingChunk] = []

    async def own(chunk: StreamingChunk) -> None:
        mine.append(chunk)

    ticket = await start_stream(
        client, pipe, {"agent": {"messages": QUESTION, "streaming_callback": own}}, protocol="ai-sdk"
    )
    stream_id = str(ticket["streamId"])
    await finished(recorders, stream_id)
    message = verify_ai_sdk(recorders[stream_id].data)
    assert message.texts == ["Let me check.", "It is sunny in Paris."]
    assert mine  # teed with the component's own callback

    with pytest.raises(ValueError, match="nested form"):
        await start_stream(client, pipe, {"messages": QUESTION})


async def test_an_agent_on_a_sync_only_generator_gets_the_sync_callback(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    release.set()
    agent = slow_agent(release, SyncOnlyChatGenerator(weather_script()))
    ticket = await start_stream(client, agent, {"messages": QUESTION})
    stream_id = str(ticket["streamId"])
    await finished(recorders, stream_id)
    assert recorders[stream_id].outcome == "end"
    assert verify_agui(recorders[stream_id].data).texts[-1] == "It is sunny in Paris."


async def test_an_agent_whose_generators_all_have_run_async_gets_the_async_callback(
    client: IQClient, recorders: dict[str, RecordingPublisher], caplog: pytest.LogCaptureFixture
) -> None:
    release = threading.Event()
    release.set()
    agent = slow_agent(release, FallbackChatGenerator([ScriptedChatGenerator(weather_script())]))
    with caplog.at_level(logging.WARNING):
        ticket = await start_stream(client, agent, {"messages": QUESTION})
        await finished(recorders, str(ticket["streamId"]))
    assert recorders[str(ticket["streamId"])].outcome == "end"
    assert not [record for record in caplog.records if "sync streaming callback" in record.getMessage()]


def sub_agent_coordinator() -> Agent:
    assert AgentTool is not None
    sub = Agent(chat_generator=SyncOnlyChatGenerator([text_turn(["Sub-agent: ", "sunny."])]))
    research = AgentTool(agent=sub, name="research", description="Research a question")
    call = ToolCall(tool_name="research", arguments={"messages": [{"role": "user", "content": "q"}]}, id="r1")
    return Agent(
        chat_generator=ScriptedChatGenerator([tool_turn([call]), text_turn(["Done."])]),
        tools=[research],
        tool_streaming_callback_passthrough=True,
    )


@pytest.mark.parametrize("reaches", ["fallback", "sub-agent"])
async def test_an_agent_reaching_a_generator_without_run_async_gets_the_sync_callback(
    client: IQClient, recorders: dict[str, RecordingPublisher], reaches: str
) -> None:
    if reaches == "fallback":
        release = threading.Event()
        release.set()
        agent = slow_agent(release, FallbackChatGenerator([SyncOnlyChatGenerator(weather_script())]))
    elif AgentTool is None:
        pytest.skip("this haystack-ai has no AgentTool")
    else:
        agent = sub_agent_coordinator()
    saved: list[dict[str, Any]] = []
    ticket = await start_stream(
        client, agent, {"messages": QUESTION}, before_close=lambda _, result: saved.append(result)
    )
    stream_id = str(ticket["streamId"])
    await finished(recorders, stream_id)
    assert recorders[stream_id].outcome == "end" and len(saved) == 1
    conversation = verify_agui(recorders[stream_id].data)
    if reaches == "fallback":
        assert conversation.texts[-1] == "It is sunny in Paris."
    else:  # the sub-agent answered, rather than failing on a coroutine callback
        assert conversation.results == {"r1": "Sub-agent: sunny."}
        assert conversation.texts == ["Sub-agent: sunny.", "Done."]


def test_start_stream_sync_runs_in_a_thread_and_saves_before_the_close(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    saved: list[object] = []
    done = threading.Event()

    def before_close(stream_id: str, result: dict[str, Any]) -> None:
        saved.append(
            (recorders[stream_id].outcome, result["agent"]["last_message"].text, threading.current_thread().name)
        )
        done.set()

    pipe = Pipeline()
    pipe.add_component("agent", slow_agent(release))
    ticket = start_stream_sync(client, pipe, {"agent": {"messages": QUESTION}}, before_close=before_close)
    stream_id = str(ticket["streamId"])
    assert saved == []
    release.set()
    assert done.wait(5)
    deadline = time.monotonic() + 5
    while recorders[stream_id].outcome is None and time.monotonic() < deadline:
        time.sleep(0.01)
    assert saved == [(None, "It is sunny in Paris.", f"iq-stream-haystack {stream_id}")]
    assert recorders[stream_id].outcome == "end"
    assert verify_agui(recorders[stream_id].data).results == {"call_1": "Sunny in Paris"}


def test_start_stream_sync_with_an_agent_tees_its_own_callback(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    release.set()
    mine: list[str] = []
    ticket = start_stream_sync(
        client, slow_agent(release), {"messages": QUESTION, "streaming_callback": lambda c: mine.append(c.content)}
    )
    stream_id = str(ticket["streamId"])
    deadline = time.monotonic() + 5
    while (stream_id not in recorders or recorders[stream_id].outcome is None) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert recorders[stream_id].outcome == "end"
    assert "".join(mine).startswith("Let me check.")


async def test_a_passthrough_agent_given_its_tools_at_run_time_gets_the_sync_callback(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    if AgentTool is None:
        pytest.skip("this haystack-ai has no AgentTool")
    sub = Agent(chat_generator=SyncOnlyChatGenerator([text_turn(["Sub-agent: ", "sunny."])]))
    research = AgentTool(agent=sub, name="research", description="Research a question")
    call = ToolCall(tool_name="research", arguments={"messages": [{"role": "user", "content": "q"}]}, id="r1")
    agent = Agent(
        chat_generator=ScriptedChatGenerator([tool_turn([call]), text_turn(["Done."])]),
        tool_streaming_callback_passthrough=True,
    )
    saved: list[dict[str, Any]] = []
    ticket = await start_stream(
        client, agent, {"messages": QUESTION, "tools": [research]}, before_close=lambda _, result: saved.append(result)
    )
    stream_id = str(ticket["streamId"])
    await finished(recorders, stream_id)
    assert recorders[stream_id].outcome == "end" and len(saved) == 1
    # The sub-agent answered, rather than failing on a coroutine callback.
    assert verify_agui(recorders[stream_id].data).results == {"r1": "Sub-agent: sunny."}


class GivesUpRaising(RecordingPublisher):
    """A publisher of a client opened with on_error="raise", which gives up at its fourth frame."""

    def __init__(self, stream_id: str) -> None:
        super().__init__(stream_id, on_error="raise")

    def send(self, data: str, event: str | None = None) -> bool:
        if self.frames_sent == 3 and self.error is None:
            self.fail()
        return super().send(data, event)


@pytest.fixture
def raising(monkeypatch: pytest.MonkeyPatch) -> list[GivesUpRaising]:
    """The publishers a client opened with on_error="raise" would have opened, giving up mid-run."""
    opened: list[GivesUpRaising] = []

    def publisher(self: IQClient, stream_id: str) -> GivesUpRaising:
        opened.append(GivesUpRaising(stream_id))
        return opened[-1]

    monkeypatch.setattr(IQClient, "publisher", publisher)
    return opened


async def test_a_publishing_failure_of_a_completed_run_is_not_logged_as_a_failed_run(
    client: IQClient, raising: list[GivesUpRaising], caplog: pytest.LogCaptureFixture
) -> None:
    opened = raising
    release = threading.Event()
    release.set()
    saved: list[str] = []
    with caplog.at_level(logging.WARNING, "haystack_integrations.iq_stream"):
        await start_stream(
            client, slow_agent(release), {"messages": QUESTION}, before_close=lambda i, _: saved.append(i)
        )
        await wait_for(lambda: bool(opened) and opened[0].close_calls > 0)
        await wait_for(lambda: not hayhooks._TASKS)  # pyright: ignore[reportPrivateUsage]
    assert saved == [opened[0].stream_id]
    messages = [record.getMessage() for record in caplog.records]
    assert not any("the run failed" in message for message in messages)
    assert messages == [f"{opened[0].error}; the run itself completed, and before_close ran."]


def test_start_stream_sync_tells_a_publishing_failure_from_a_failed_run(
    client: IQClient, raising: list[GivesUpRaising], caplog: pytest.LogCaptureFixture
) -> None:
    opened = raising
    release = threading.Event()
    release.set()
    with caplog.at_level(logging.WARNING, "haystack_integrations.iq_stream"):
        ticket = start_stream_sync(client, slow_agent(release), {"messages": QUESTION})
        name = f"iq-stream-haystack {ticket['streamId']}"
        for thread in [thread for thread in threading.enumerate() if thread.name == name]:
            thread.join(5)
    assert [record.getMessage() for record in caplog.records] == [
        f"{opened[0].error}; the run itself completed, and before_close ran."
    ]
