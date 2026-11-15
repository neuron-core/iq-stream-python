"""IQStream around real Haystack Agents on a scripted model: close lines, persistence, fail-open, sizes, errors."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import re
import threading
import time
from collections.abc import Hashable
from pathlib import Path
from typing import Any

import pytest
from haystack.components.agents import Agent
from haystack.dataclasses import ChatMessage, StreamingChunk
from haystack.hooks.human_in_the_loop import ConfirmationHook, ToolExecutionDecision
from haystack.tools import tool
from iq_stream import IQClient, IQError, Outcome, PublishResult, frame_size
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher

from haystack_integrations.iq_stream import AGUIConverter, Frame, IQStream
from tests.checks import verify_agui, verify_ai_sdk
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


@pytest.mark.parametrize(("close_on_cancel", "outcome"), [("fail", "fail"), ("end", "end")])
async def test_a_cancelled_run_finishes_cancelled_and_closes_as_configured(close_on_cancel: str, outcome: str) -> None:
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
        async with IQStream(publisher, close_on_cancel=close_on_cancel) as stream:  # pyright: ignore[reportArgumentType]
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


class GivesUp(RecordingPublisher):
    """Gives up at its fourth frame, as a publisher whose server went away or whose stream grew too large."""

    def send(self, data: str, event: str | None = None) -> bool:
        if self.frames_sent == 3 and self.error is None:
            self.fail()
        return super().send(data, event)


def test_a_publisher_that_gave_up_never_breaks_the_run_which_is_read_to_its_end(
    caplog: pytest.LogCaptureFixture,
) -> None:
    publisher = RecordingPublisher()
    publisher.fail()
    reported: list[Exception] = []
    generator = ScriptedChatGenerator(weather_script())
    agent = Agent(chat_generator=generator, tools=[get_weather])
    with caplog.at_level(logging.WARNING), IQStream(publisher, on_dropped=reported.append) as stream:
        result = agent.run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert generator.calls == 2
    assert publisher.data == [] and stream.failed
    assert isinstance(stream.error, IQError)
    assert reported == [publisher.error]  # on_dropped hears of the core's error, once
    assert not [r for r in caplog.records if r.name == "haystack_integrations.iq_stream"]  # the core logged it


def test_a_publisher_that_gives_up_mid_run_stops_publishing_and_the_run_goes_on() -> None:
    publisher = GivesUp()
    reported: list[Exception] = []
    with IQStream(publisher, on_dropped=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
        assert reported == [publisher.error]  # as soon as the adapter saw it, before the block was left
    same_run(result)
    assert len(publisher.data) == 3 and stream.failed
    assert reported == [publisher.error]


def test_a_publisher_that_raises_is_reported_once_and_the_stream_closes_with_fail() -> None:
    class Raises(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            if self.frames_sent == 2:
                raise RuntimeError("socket exploded")
            return super().send(data, event)

    publisher = Raises()
    reported: list[IQError] = []
    with IQStream(publisher, on_dropped=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert [(error.code, str(error.__cause__)) for error in reported] == [("publish_failed", "socket exploded")]
    assert publisher.outcome == "fail" and stream.failed and len(publisher.data) == 2
    assert stream.error is reported[0]


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
    assert publisher.outcome == "end" and not s.failed  # every chunk was skipped alone; the run completed


def test_a_chunk_that_cannot_be_converted_is_skipped_and_the_rest_of_the_run_is_published() -> None:
    class BreaksOnce(AGUIConverter):
        chunks = 0

        def convert(self, chunk: StreamingChunk, source: Hashable) -> list[Frame]:
            self.chunks += 1
            if self.chunks == 2:
                raise ValueError("unexpected chunk")
            return super().convert(chunk, source)

    publisher = RecordingPublisher()
    reported: list[IQError] = []
    converter = BreaksOnce(thread_id="thread-1", run_id="run-1")
    with IQStream(publisher, protocol=converter, on_dropped=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert [error.code for error in reported] == ["encode_failed"]
    assert isinstance(reported[0].__cause__, ValueError) and "unexpected" not in str(reported[0])
    assert publisher.outcome == "end" and not stream.failed
    conversation = verify_agui(publisher.data)
    assert conversation.texts == ["Let check.", "It is sunny in Paris."]  # only the chunk "me " is missing
    assert conversation.results == {"call_1": "Sunny, 21C in Paris"}
    assert conversation.terminal["type"] == "RUN_FINISHED"


def test_with_on_error_raise_the_run_is_read_to_its_end_and_leaving_the_block_raises() -> None:
    publisher = GivesUp(on_error="raise")
    reported: list[Exception] = []
    results: list[dict[str, Any]] = []
    with pytest.raises(IQError) as raised, IQStream(publisher, on_dropped=reported.append) as stream:
        results.append(weather_agent().run(QUESTION, streaming_callback=stream.callback))
        assert stream.custom("progress", 1) is False  # nothing raises into the run
    same_run(results[0])
    assert raised.value is publisher.error and reported == [publisher.error]
    assert publisher.close_calls == 1 and len(publisher.data) == 3


def test_with_on_error_raise_a_publisher_dead_from_the_start_still_lets_the_run_complete() -> None:
    publisher = RecordingPublisher(on_error="raise")
    publisher.fail()
    results: list[dict[str, Any]] = []
    with pytest.raises(IQError), IQStream(publisher) as stream:
        results.append(weather_agent().run(QUESTION, streaming_callback=stream.callback))
    same_run(results[0])


def test_with_on_error_raise_the_block_s_own_exception_is_the_one_that_goes_on() -> None:
    publisher = GivesUp(on_error="raise")
    with pytest.raises(OSError, match="disk full"), IQStream(publisher) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
        raise OSError("disk full")
    assert publisher.close_calls == 1


def adapter_logs(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "haystack_integrations.iq_stream"]


def test_with_on_error_raise_a_give_up_is_raised_and_never_logged_by_the_adapter(
    caplog: pytest.LogCaptureFixture,
) -> None:
    publisher = GivesUp(on_error="raise")
    with caplog.at_level(logging.WARNING), pytest.raises(IQError) as raised, IQStream(publisher) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
    assert raised.value is publisher.error and adapter_logs(caplog) == []


def test_with_on_error_raise_a_give_up_next_to_the_block_s_exception_is_logged_once(
    caplog: pytest.LogCaptureFixture,
) -> None:
    publisher = GivesUp(on_error="raise")
    with caplog.at_level(logging.WARNING), pytest.raises(OSError, match="disk full"), IQStream(publisher) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
        raise OSError("disk full")
    # The core logs only in report mode, and nothing raised it: the adapter says it once, with the core's text.
    assert adapter_logs(caplog) == [f"{publisher.error}; the block's own exception goes on."]
    assert stream.failed and publisher.close_calls == 1


def test_in_report_mode_the_core_s_give_up_is_left_to_the_core_s_log(caplog: pytest.LogCaptureFixture) -> None:
    publisher = GivesUp()
    with caplog.at_level(logging.WARNING), pytest.raises(OSError), IQStream(publisher) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
        raise OSError("disk full")
    assert stream.failed and adapter_logs(caplog) == []


def test_a_publisher_that_cannot_open_is_reported_and_the_run_goes_on(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(self: IQClient, stream_id: str) -> Any:
        raise RuntimeError("can't start new thread")

    monkeypatch.setattr(IQClient, "publisher", refuse)
    client = IQClient("http://127.0.0.1:9", "demo-app", "test-app-secret-0123456789abcdef0123")
    reported: list[IQError] = []
    with IQStream(client, on_dropped=reported.append) as stream:
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert stream.failed and stream.result is None
    assert [(e.code, str(e.__cause__)) for e in reported] == [("publish_failed", "can't start new thread")]


def test_an_on_dropped_that_raises_is_logged_and_contained(caplog: pytest.LogCaptureFixture) -> None:
    class Raises(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            raise RuntimeError("nope")

    def handler(error: Exception) -> None:
        raise KeyError("handler bug")

    with caplog.at_level(logging.ERROR), IQStream(Raises(), on_dropped=handler) as stream:
        same_run(weather_agent().run(QUESTION, streaming_callback=stream.callback))
    assert any("on_dropped raised" in record.getMessage() for record in caplog.records)


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
    # What the browser rebuilds from the pieces is the whole text, reasoning and arguments.
    texts, thinking, arguments = ["word " * 2_000, "word " * 2_000], ["think " * 1_000], {"city": "€" * 3_000}
    marker = "[truncated, 900000 characters in total]"
    if protocol == "ag-ui":
        conversation = verify_agui(publisher.data)
        assert (conversation.texts, conversation.reasoning) == (texts, thinking)
        assert json.loads(conversation.calls[0][2]) == arguments
        assert conversation.results["c1"].endswith(marker)
    elif protocol == "ai-sdk":
        message = verify_ai_sdk(publisher.data)
        assert (message.texts, message.reasoning) == (texts, thinking)
        assert json.loads("".join(message.tools["c1"]["deltas"])) == arguments
        assert message.tools["c1"]["output"].endswith(marker)
    else:
        chunks = [chunk for chunk in events(publisher) if chunk["type"] == "chunk"]
        assert "".join(chunk["content"] for chunk in chunks) == "".join(texts)
        assert "".join(chunk["reasoning"]["reasoning_text"] for chunk in chunks if chunk["reasoning"]) == thinking[0]
        deltas = [delta for chunk in chunks for delta in chunk["tool_calls"] or []]
        assert json.loads("".join(delta["arguments"] or "" for delta in deltas)) == arguments
        result = next(chunk["tool_call_result"] for chunk in chunks if chunk["tool_call_result"])
        assert result["result"].endswith(marker)


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
        assert stream.emit("x" * 2_000) is False  # dropped, and the stream goes on
        assert stream.custom("progress", {"done": 1})
    assert publisher.data[1:5] == [
        '{"type":"STATE_SNAPSHOT","snapshot":{"step":1}}',
        '{"type":"CUSTOM","name":"note","value":"é"}',
        "plain",
        '{"type":"CUSTOM","name":"progress","value":{"done":1}}',
    ]
    assert publisher.frames[3].event == "note"
    assert stream.emit("too late") is False


@pytest.mark.parametrize("protocol", ["ag-ui", "ai-sdk", "haystack"])
def test_custom_and_emit_never_raise_into_the_run_and_drop_only_a_frame_too_large(protocol: str) -> None:
    publisher = RecordingPublisher(max_frame_bytes=1024)
    reported: list[Exception] = []
    with IQStream(publisher, protocol=protocol, on_dropped=reported.append) as stream:  # pyright: ignore[reportArgumentType]
        assert stream.custom("note", {"text": "x" * 2_000}) is False
        assert stream.emit({"text": "x" * 2_000}) is False
        assert stream.custom("note", {"text": "small"})
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
    assert [type(error).__name__ for error in reported] == ["FrameTooLarge", "FrameTooLarge"]
    assert not stream.failed and publisher.outcome == "end"
    assert sum('"small"' in item for item in publisher.data) == 1
    assert not any("x" * 1_000 in item for item in publisher.data)


def test_custom_through_a_publisher_that_raises_stops_publishing_and_the_run_goes_on() -> None:
    class Raises(RecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            if "CUSTOM" in data:
                raise RuntimeError("socket exploded")
            return super().send(data, event)

    reported: list[IQError] = []
    with IQStream(Raises(), on_dropped=reported.append) as stream:
        assert stream.custom("tool_confirmation", {"tool": "get_weather"}) is False
        same_run(weather_agent().run(QUESTION, streaming_callback=stream.callback))
    assert [str(error.__cause__) for error in reported] == ["socket exploded"]
    assert stream.failed and stream.publisher is not None
    assert stream.publisher.outcome == "fail"  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.parametrize("protocol", ["ag-ui", "ai-sdk", "haystack"])
def test_custom_writes_what_json_cannot_hold_as_its_str_in_every_protocol(protocol: str) -> None:
    class Ticket:
        def __str__(self) -> str:
            return "ticket-1"

    publisher = RecordingPublisher()
    with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        assert stream.custom("confirm", {"ticket": Ticket(), "at": (1, 2)})
    assert '{"ticket":"ticket-1","at":[1,2]}' in publisher.data[1]


def test_emit_never_raises_into_the_run_whatever_its_payload_holds() -> None:
    from ag_ui.core import CustomEvent

    class Ticket:
        def __str__(self) -> str:
            return "ticket-1"

    publisher = RecordingPublisher()
    reported: list[IQError] = []
    with IQStream(publisher, on_dropped=reported.append) as stream:
        assert stream.emit(CustomEvent(name="confirm", value={"ticket": Ticket()}))  # written as its str()
        assert stream.emit({("a", 1): "a key JSON cannot hold"}) is False  # dropped and reported
        assert stream.emit("bad name", event="not a name!") is False
        cycle: dict[str, object] = {}
        cycle["self"] = cycle
        assert stream.emit(cycle) is False
        result = weather_agent().run(QUESTION, streaming_callback=stream.callback)
    same_run(result)
    assert publisher.data[1] == '{"type":"CUSTOM","name":"confirm","value":{"ticket":"ticket-1"}}'
    assert [(error.code, type(error.__cause__).__name__) for error in reported] == [
        ("encode_failed", "TypeError"),
        ("encode_failed", "ValueError"),
        ("encode_failed", "ValueError"),
    ]
    assert publisher.outcome == "end" and not stream.failed


@pytest.mark.parametrize("protocol", ["ag-ui", "ai-sdk", "haystack"])
def test_a_nan_or_an_infinity_is_written_as_null_in_every_protocol(protocol: str) -> None:
    from ag_ui.core import CustomEvent

    def strict(data: str) -> Any:
        return json.loads(data, parse_constant=lambda name: pytest.fail(f"{name} is not JSON"))

    publisher = RecordingPublisher()
    with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        assert stream.custom("score", {"confidence": math.nan, "range": [-math.inf, math.inf], "n": 1.5})
        assert stream.emit({"score": math.nan})
        assert stream.emit(CustomEvent(name="score", value=math.inf))
    frames = [strict(data) for data in publisher.data[1:4]]
    custom = frames[0]["value"] if protocol != "ai-sdk" else frames[0]["data"]
    assert custom == {"confidence": None, "range": [None, None], "n": 1.5}
    assert frames[1] == {"score": None} and frames[2]["value"] is None


def test_custom_sends_nothing_unless_every_frame_of_it_fits() -> None:
    class TwoFrames(AGUIConverter):
        def custom(self, name: str, value: object) -> list[Frame]:
            return [Frame('{"type":"CUSTOM","name":"part","value":1}'), Frame("x" * 2_000)]

    publisher = RecordingPublisher(max_frame_bytes=1024)
    reported: list[IQError] = []
    converter = TwoFrames(thread_id="thread-1", run_id="run-1", max_frame_bytes=1024)
    with IQStream(publisher, protocol=converter, on_dropped=reported.append) as stream:
        assert stream.custom("note", None) is False
    assert not any("part" in data for data in publisher.data)
    assert [type(error).__name__ for error in reported] == ["FrameTooLarge"]
    assert publisher.outcome == "end"


def test_a_publisher_dead_on_entry_is_reported_before_the_block_runs_even_with_no_start_event() -> None:
    class Silent:
        def start(self) -> list[Frame]:
            return []

        def convert(self, chunk: StreamingChunk, source: object) -> list[Frame]:
            return []

        def finish(self) -> list[Frame]:
            return []

        def error(self, exc: BaseException) -> list[Frame]:
            return []

        cancelled = finish

        def custom(self, name: str, value: object) -> list[Frame]:
            return []

    publisher = RecordingPublisher()
    publisher.fail()
    reported: list[IQError] = []
    with IQStream(publisher, protocol=Silent(), on_dropped=reported.append):
        assert reported == [publisher.error]
    assert reported == [publisher.error]


def test_the_readme_s_confirmation_strategy_survives_arguments_too_large_for_a_frame() -> None:
    readme = (Path(__file__).resolve().parents[2] / "README.md").read_text(encoding="utf-8")
    strategy = re.search(r"^class AskTheBrowser:\n.*?(?=^hook = )", readme, re.DOTALL | re.MULTILINE)
    assert strategy is not None
    namespace: dict[str, Any] = {"wait_for_answer": lambda tool_call_id: True}
    exec("from typing import Any\nfrom haystack.hooks.human_in_the_loop import ToolExecutionDecision\n"
         "from haystack_integrations.iq_stream import IQStream\n" + strategy.group(), namespace)  # fmt: skip

    @tool
    def save_note(text: str) -> str:
        """Saves a note."""
        return "Saved."

    note = weather_call().__class__(tool_name="save_note", arguments={"text": "n" * 2_000}, id="c1")
    script = [tool_turn([note]), text_turn(["Done."])]
    hook = ConfirmationHook(confirmation_strategies={"save_note": namespace["AskTheBrowser"]()})
    agent = Agent(chat_generator=ScriptedChatGenerator(script), tools=[save_note], hooks={"before_tool": [hook]})
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with IQStream(publisher, on_dropped=lambda error: None) as stream:
        result = agent.run(QUESTION, streaming_callback=stream.callback, hook_context={"stream": stream})
    assert result["last_message"].text == "Done."
    conversation = verify_agui(publisher.data)
    assert conversation.custom == [("tool_confirmation", {"toolCallId": "c1", "tool": "save_note", "arguments": None})]
    assert conversation.results == {"c1": "Saved."}
    assert publisher.outcome == "end" and not stream.failed


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
        {"close_on_cancel": "maybe"},
        {"max_tool_result_bytes": 0},
        {"close_timeout": -1},
        {"close_timeout": 0},
        {"close_timeout": math.inf},
        {"close_timeout": math.nan},
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


def test_an_asynchronous_publisher_is_refused_since_haystack_calls_back_from_threads() -> None:
    with pytest.raises(TypeError, match="synchronous publisher"):
        IQStream(AsyncRecordingPublisher())  # pyright: ignore[reportArgumentType]


def test_a_close_that_raises_is_retried_with_fail_so_the_stream_is_never_left_open() -> None:
    class CloseBreaksOnce(RecordingPublisher):
        attempts = 0

        def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
            self.attempts += 1
            if self.attempts == 1:
                raise RuntimeError("close broke")
            return super().close(outcome, timeout)

    publisher = CloseBreaksOnce()
    reported: list[IQError] = []
    with IQStream(publisher, on_dropped=reported.append) as stream:
        weather_agent().run(QUESTION, streaming_callback=stream.callback)
    assert publisher.outcome == "fail" and not publisher.alive
    assert [str(error.__cause__) for error in reported] == ["close broke"]
    assert stream.result is not None and stream.result.outcome == "fail"


async def test_leaving_an_async_block_closes_off_the_event_loop() -> None:
    class SlowClose(RecordingPublisher):
        def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
            time.sleep(1.0)  # as the core waits out min_open, then for the server's answer
            return super().close(outcome, timeout)

    gaps: list[float] = []

    async def heartbeat() -> None:
        last = time.monotonic()
        while True:
            await asyncio.sleep(0.05)
            gaps.append(time.monotonic() - last)
            last = time.monotonic()

    beat = asyncio.create_task(heartbeat())
    publisher = SlowClose()
    async with IQStream(publisher) as stream:
        await weather_agent().run_async(QUESTION, streaming_callback=stream.acallback)
    beat.cancel()
    assert publisher.outcome == "end"
    assert max(gaps) < 0.5, max(gaps)  # the loop kept running while the close waited a second
