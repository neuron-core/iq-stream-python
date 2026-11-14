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
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.tools import tool
from iq_stream import IQClient
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import hayhooks
from haystack_integrations.iq_stream.hayhooks import start_stream, start_stream_sync
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import ScriptedChatGenerator, SyncOnlyChatGenerator, Turn, text_turn, weather_script

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


async def test_start_stream_answers_at_once_saves_before_the_close_and_keeps_the_task(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    saved: list[tuple[str, object]] = []

    async def on_result(stream_id: str, result: dict[str, Any]) -> None:
        saved.append((stream_id, recorders[stream_id].outcome))

    ticket = await start_stream(client, slow_agent(release), {"messages": QUESTION}, on_result=on_result, run_id="r1")
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
        await wait_for(lambda: stream_id in recorders and recorders[stream_id].outcome is not None)
        await wait_for(lambda: not hayhooks._TASKS)  # pyright: ignore[reportPrivateUsage]
    assert recorders[stream_id].outcome == "fail"
    assert any("the run failed" in record.getMessage() for record in caplog.records)


async def test_a_save_that_fails_closes_with_fail(client: IQClient, recorders: dict[str, RecordingPublisher]) -> None:
    def on_result(stream_id: str, result: dict[str, Any]) -> None:
        raise OSError("database down")

    release = threading.Event()
    release.set()
    ticket = await start_stream(client, slow_agent(release), {"messages": QUESTION}, on_result=on_result)
    stream_id = str(ticket["streamId"])
    await wait_for(lambda: stream_id in recorders and recorders[stream_id].outcome is not None)
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
    await wait_for(lambda: stream_id in recorders and recorders[stream_id].outcome is not None)
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
    await wait_for(lambda: stream_id in recorders and recorders[stream_id].outcome is not None)
    assert recorders[stream_id].outcome == "end"
    assert verify_agui(recorders[stream_id].data).texts[-1] == "It is sunny in Paris."


def test_start_stream_sync_runs_in_a_thread_and_saves_before_the_close(
    client: IQClient, recorders: dict[str, RecordingPublisher]
) -> None:
    release = threading.Event()
    saved: list[object] = []
    done = threading.Event()

    def on_result(stream_id: str, result: dict[str, Any]) -> None:
        saved.append(
            (recorders[stream_id].outcome, result["agent"]["last_message"].text, threading.current_thread().name)
        )
        done.set()

    pipe = Pipeline()
    pipe.add_component("agent", slow_agent(release))
    ticket = start_stream_sync(client, pipe, {"agent": {"messages": QUESTION}}, on_result=on_result)
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
