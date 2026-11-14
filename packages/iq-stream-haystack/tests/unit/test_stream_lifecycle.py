"""IQStream around real Haystack Agents on a scripted model: close lines, persistence, fail-open, sizes, errors."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import pytest
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.hooks.human_in_the_loop import ConfirmationHook, ToolExecutionDecision
from haystack.tools import tool
from iq_stream import FrameTooLarge, IQClient, IQError
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import IQStream
from haystack_integrations.iq_stream.converters.base import frame_size
from tests.checks import verify_agui
from tests.fakes import ScriptedChatGenerator, Turn, text_turn, tool_turn, weather_call, weather_script

QUESTION = [ChatMessage.from_user("Weather in Paris?")]


@tool
def get_weather(city: str) -> str:
    """Get the weather for a city."""
    return f"Sunny, 21C in {city}"


def weather_agent(**options: Any) -> Agent:
    return Agent(chat_generator=ScriptedChatGenerator(weather_script()), tools=[get_weather], **options)


def events(publisher: RecordingPublisher) -> list[dict[str, Any]]:
    return [json.loads(item) for item in publisher.data]


# -- close lines ---------------------------------------------------------------------------------------------------


def test_a_completed_run_ends_with_run_finished_and_end() -> None:
    publisher = RecordingPublisher()
    with IQStream(publisher, run_id="run-1", thread_id="chat-1") as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    assert publisher.outcome == "end"
    conversation = verify_agui(publisher.data)
    assert conversation.terminal == {"type": "RUN_FINISHED", "threadId": "chat-1", "runId": "run-1"}
    assert conversation.texts == ["Let me check.", "It is sunny in Paris."]
    assert conversation.results == {"call_1": "Sunny, 21C in Paris"}
    assert result["last_message"].text == "It is sunny in Paris."
    assert stream.result is not None and stream.result.outcome == "end"
    assert not stream.failed


def test_a_failing_run_ends_with_a_masked_run_error_and_fail_and_the_exception_goes_on() -> None:
    error = RuntimeError("provider said: key sk-secret is invalid")
    script = [Turn(text_turn(["Hel", "lo"]).chunks, ChatMessage.from_assistant("Hello"), error=error, fail_after=1)]
    agent = Agent(chat_generator=ScriptedChatGenerator(script))
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError) as raised, IQStream(publisher) as stream:
        agent.run(QUESTION, streaming_callback=stream.callback)
    assert raised.value is error
    assert publisher.outcome == "fail"
    conversation = verify_agui(publisher.data)
    assert conversation.terminal == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "error"}
    assert conversation.texts == ["Hel"]  # the open message was closed before the error
    assert "sk-secret" not in "".join(publisher.data)


def test_a_failing_tool_with_raise_on_failure_fails_the_stream_with_exposed_text_when_asked() -> None:
    @tool
    def broken(city: str) -> str:
        """Always fails."""
        raise ConnectionError("weather service is down")

    script = [tool_turn([weather_call().__class__(tool_name="broken", arguments={"city": "Paris"}, id="c1")])]
    agent = Agent(chat_generator=ScriptedChatGenerator(script), tools=[broken], raise_on_tool_invocation_failure=True)
    publisher = RecordingPublisher()
    with pytest.raises(Exception, match="weather service is down"), IQStream(publisher, expose_errors=True) as s:
        agent.run(QUESTION, streaming_callback=s.callback)
    assert publisher.outcome == "fail"
    terminal = events(publisher)[-1]
    assert terminal["type"] == "RUN_ERROR" and "weather service is down" in terminal["message"]


def test_a_failing_tool_without_raise_on_failure_is_an_error_result_and_the_run_ends() -> None:
    @tool
    def broken(city: str) -> str:
        """Always fails."""
        raise ConnectionError("internal host db-7 refused")

    call = weather_call().__class__(tool_name="broken", arguments={"city": "Paris"}, id="c1")
    agent = Agent(chat_generator=ScriptedChatGenerator([tool_turn([call]), text_turn(["Sorry."])]), tools=[broken])
    publisher = RecordingPublisher()
    with IQStream(publisher) as stream:
        agent.run(QUESTION, streaming_callback=stream.callback)
    assert publisher.outcome == "end"
    assert verify_agui(publisher.data).results == {"c1": "An error occurred."}
    assert "db-7" not in "".join(publisher.data)


@pytest.mark.parametrize(("cancelled_close", "outcome"), [("fail", "fail"), ("end", "end")])
async def test_a_cancelled_run_finishes_cancelled_and_closes_as_configured(cancelled_close: str, outcome: str) -> None:
    started = asyncio.Event()

    @tool
    async def slow_weather(city: str) -> str:
        """Takes its time."""
        started.set()
        await asyncio.sleep(30)
        return "never"

    call = weather_call().__class__(tool_name="slow_weather", arguments={"city": "Paris"}, id="c1")
    agent = Agent(chat_generator=ScriptedChatGenerator([tool_turn([call]), text_turn(["x"])]), tools=[slow_weather])
    publisher = RecordingPublisher()

    async def run() -> None:
        async with IQStream(publisher, cancelled_close=cancelled_close) as stream:  # pyright: ignore[reportArgumentType]
            await agent.run_async(QUESTION, streaming_callback=stream.acallback)

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.outcome == outcome
    conversation = verify_agui(publisher.data)
    assert conversation.terminal["outcome"] == {"type": "cancelled"}


def test_a_human_in_the_loop_pause_keeps_the_stream_open_and_the_run_ends_after_the_answer() -> None:
    answered = threading.Event()

    class AskTheBrowser:
        """Publishes the request through the stream in hook_context, then blocks until the user answers."""

        def run(
            self,
            *,
            tool_name: str,
            tool_description: str,
            tool_params: dict[str, Any],
            tool_call_id: str | None = None,
            confirmation_strategy_context: dict[str, Any] | None = None,
        ) -> ToolExecutionDecision:
            assert confirmation_strategy_context is not None
            stream: IQStream = confirmation_strategy_context["stream"]
            stream.custom(
                "tool_confirmation", {"toolCallId": tool_call_id, "tool": tool_name, "arguments": tool_params}
            )
            assert answered.wait(5)
            return ToolExecutionDecision(
                tool_name=tool_name, execute=True, tool_call_id=tool_call_id, final_tool_params=tool_params
            )

        async def run_async(self, **kwargs: Any) -> ToolExecutionDecision:
            return self.run(**kwargs)

    hook = ConfirmationHook(confirmation_strategies={"get_weather": AskTheBrowser()})  # pyright: ignore[reportArgumentType]
    agent = weather_agent(hooks={"before_tool": [hook]})
    publisher = RecordingPublisher()
    done = threading.Event()

    def run() -> None:
        with IQStream(publisher) as stream:
            agent.run(QUESTION, streaming_callback=stream.callback, hook_context={"stream": stream})
        done.set()

    worker = threading.Thread(target=run)
    worker.start()
    deadline = time.monotonic() + 5
    while not any('"CUSTOM"' in item for item in publisher.data) and time.monotonic() < deadline:
        time.sleep(0.01)
    # Paused: the request is out, the tool has not run, and the stream is open.
    assert events(publisher)[-1]["type"] == "CUSTOM"
    assert publisher.outcome is None and not done.is_set()
    answered.set()
    worker.join(5)
    assert publisher.outcome == "end"
    conversation = verify_agui(publisher.data)
    assert conversation.custom == [
        ("tool_confirmation", {"toolCallId": "call_1", "tool": "get_weather", "arguments": {"city": "Paris"}})
    ]
    assert conversation.results == {"call_1": "Sunny, 21C in Paris"}


def test_the_answer_is_saved_before_the_close_line() -> None:
    publisher = RecordingPublisher()
    seen: list[object] = []

    def save(messages: list[ChatMessage]) -> None:
        seen.append((publisher.outcome, publisher.close_calls, events(publisher)[-1]["type"], len(messages)))

    with IQStream(publisher) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
        save(result["messages"])
    assert seen == [(None, 0, "TEXT_MESSAGE_END", 4)]
    assert publisher.outcome == "end" and events(publisher)[-1]["type"] == "RUN_FINISHED"


def test_a_save_that_raises_fails_the_stream() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(OSError, match="disk full"), IQStream(publisher) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
        raise OSError("disk full")
    assert publisher.outcome == "fail" and events(publisher)[-1]["type"] == "RUN_ERROR"


# -- fail open -----------------------------------------------------------------------------------------------------


def baseline() -> dict[str, Any]:
    return weather_agent().run(QUESTION)


def same_run(result: dict[str, Any]) -> None:
    expected = baseline()
    assert [m.to_dict() for m in result["messages"]] == [m.to_dict() for m in expected["messages"]]


def test_a_publisher_that_gave_up_never_breaks_the_run_which_is_read_to_its_end() -> None:
    publisher = RecordingPublisher()
    publisher.fail()
    reported: list[Exception] = []
    generator = ScriptedChatGenerator(weather_script())
    agent = Agent(chat_generator=generator, tools=[get_weather])
    with IQStream(publisher, on_error=reported.append) as stream:
        result = agent.run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert generator.calls == 2
    assert publisher.data == [] and stream.failed
    assert isinstance(stream.error, IQError)
    assert reported == []  # the core reports its own errors


def test_a_publisher_that_gives_up_mid_run_stops_publishing_and_the_run_goes_on() -> None:
    class GivesUp(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            if self.frames_sent == 3:
                self.fail()
            return super().send(data, event)

    publisher = GivesUp()
    with IQStream(publisher) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert len(publisher.data) == 3 and stream.failed


def test_a_publisher_that_raises_is_reported_once_and_the_stream_closes_with_fail() -> None:
    class Raises(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            if self.frames_sent == 2:
                raise RuntimeError("socket exploded")
            return super().send(data, event)

    publisher = Raises()
    reported: list[Exception] = []
    with IQStream(publisher, on_error=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert [str(error) for error in reported] == ["socket exploded"]
    assert publisher.outcome == "fail" and stream.failed and len(publisher.data) == 2


def test_a_converter_that_raises_is_logged_once_without_data_and_the_run_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    class Broken:
        def start(self) -> list[Any]:
            return []

        def convert(self, chunk: StreamingChunk, source: object) -> list[Any]:
            raise ValueError(f"cannot convert {chunk.content!r}")

        def finish(self) -> list[Any]:
            return []

        error = cancelled = custom = finish  # pyright: ignore[reportAssignmentType]

    publisher = RecordingPublisher()
    with (
        caplog.at_level(logging.WARNING, "haystack_integrations.iq_stream"),
        IQStream(publisher, protocol=Broken()) as s,
    ):
        result = weather_agent().run(QUESTION, streaming_callback=s.callback)
    same_run(result)
    warnings = [record.getMessage() for record in caplog.records if record.name == "haystack_integrations.iq_stream"]
    assert len(warnings) == 1 and "ValueError" in warnings[0] and "Let" not in warnings[0]
    assert publisher.outcome == "fail"


def test_a_publisher_that_raises_on_every_call_is_contained() -> None:
    publisher = RecordingPublisher(on_error="raise")
    publisher.fail()
    reported: list[Exception] = []
    with IQStream(publisher, on_error=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert len(reported) == 1 and isinstance(reported[0], IQError)


def test_a_publisher_that_cannot_open_is_reported_and_the_run_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: IQClient, stream_id: str) -> Any:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(IQClient, "publisher", refuse)
    client = IQClient("http://127.0.0.1:9", "demo-app", "test-app-secret-0123456789abcdef0123")
    reported: list[Exception] = []
    with IQStream(client, on_error=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert stream.failed and stream.result is None and [str(e) for e in reported] == ["can't start new thread"]


def test_an_on_error_that_raises_is_logged_and_contained(caplog: pytest.LogCaptureFixture) -> None:
    class Raises(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            raise RuntimeError("nope")

    def handler(error: Exception) -> None:
        raise KeyError("handler bug")

    with caplog.at_level(logging.ERROR), IQStream(Raises(), on_error=handler) as stream:
        same_run(weather_agent().run(QUESTION, streaming_callback=stream.callback))
    assert any("on_error raised" in record.getMessage() for record in caplog.records)


# -- sizes ---------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("protocol", ["ag-ui", "ai-sdk", "haystack"])
def test_every_frame_fits_a_small_limit(protocol: str) -> None:
    @tool
    def big(city: str) -> str:
        """A huge result."""
        return 'ü"\n' * 300_000

    long_text = ["word " * 2_000]
    call = weather_call().__class__(tool_name="big", arguments={"city": "€" * 3_000}, id="c1")
    script = [tool_turn([call], text=long_text, reasoning=["think " * 1_000]), text_turn(long_text)]
    agent = Agent(chat_generator=ScriptedChatGenerator(script), tools=[big])
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        agent.run(QUESTION, streaming_callback=stream.callback)
    assert not stream.failed and publisher.outcome == "end"
    assert all(frame_size(item) <= 1024 for item in publisher.data)
    if protocol == "ag-ui":
        conversation = verify_agui(publisher.data)
        assert conversation.texts == ["word " * 2_000, "word " * 2_000]
        assert json.loads(conversation.calls[0][2]) == {"city": "€" * 3_000}
        assert conversation.results["c1"].endswith("[truncated, 900000 characters in total]")


# -- frames of the application's own -------------------------------------------------------------------------------


def test_emit_publishes_events_mappings_and_strings_in_order() -> None:
    from ag_ui.core import StateSnapshotEvent

    publisher = RecordingPublisher(max_frame_bytes=1024)
    stream = IQStream(publisher)
    assert stream.emit("too early") is False
    with stream:
        assert stream.emit(StateSnapshotEvent(snapshot={"step": 1}))
        assert stream.emit({"type": "CUSTOM", "name": "note", "value": "é"})
        assert stream.emit("plain", event="note")
        with pytest.raises(FrameTooLarge):
            stream.emit("x" * 2_000)
        assert stream.custom("progress", {"done": 1})
    assert publisher.data[1:5] == [
        '{"type":"STATE_SNAPSHOT","snapshot":{"step":1}}',
        '{"type":"CUSTOM","name":"note","value":"é"}',
        "plain",
        '{"type":"CUSTOM","name":"progress","value":{"done":1}}',
    ]
    assert publisher.frames[3].event == "note"
    assert stream.emit("too late") is False


# -- the object itself ---------------------------------------------------------------------------------------------


def test_creating_a_stream_opens_nothing_and_gives_the_ticket(monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr(
        IQClient, "publisher", lambda self, stream_id: opened.append(stream_id) or RecordingPublisher(stream_id)
    )  # pyright: ignore[reportUnknownLambdaType]
    client = IQClient("http://127.0.0.1:9", "demo-app", "test-app-secret-0123456789abcdef0123")
    stream = IQStream(client)
    ticket = stream.ticket()
    assert ticket.stream_id == stream.stream_id and client.verify("sub", ticket.token, stream.stream_id)
    assert opened == []
    worker = IQStream(client, stream.stream_id)
    with worker:
        assert opened == [stream.stream_id]
    with pytest.raises(RuntimeError):
        worker.__enter__()
    with pytest.raises(RuntimeError, match="IQClient"):
        IQStream(RecordingPublisher()).ticket()


@pytest.mark.parametrize(
    "options",
    [
        {"stream_id": "short"},
        {"protocol": "openai"},
        {"cancelled_close": "maybe"},
        {"max_tool_result_bytes": 0},
    ],
)
def test_invalid_options_are_refused(options: dict[str, Any]) -> None:
    client = IQClient("http://127.0.0.1:9", "demo-app", "test-app-secret-0123456789abcdef0123")
    with pytest.raises(ValueError):
        IQStream(client, **options)


def test_a_publisher_with_another_stream_id_is_refused() -> None:
    with pytest.raises(ValueError, match="differs"):
        IQStream(RecordingPublisher("a" * 22), "b" * 22)


def test_exit_returns_false_so_the_exception_is_never_swallowed() -> None:
    stream = IQStream(RecordingPublisher())
    stream.__enter__()
    assert stream.__exit__(KeyError, KeyError("x"), None) is False
