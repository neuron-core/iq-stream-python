"""The block: which close line, fail-open, persistence before the close line, the application's own frames."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import threading
from collections.abc import Iterable
from typing import Any

import pytest
from agui_checks import checked, conversation, types
from crew_fakes import ANSWER, FakeStreamingLLM, LookupTool, make_crew
from crewai import Agent
from crewai.events import crewai_event_bus
from crewai.events.base_events import BaseEvent
from crewai.events.types.crew_events import CrewKickoffStartedEvent
from iq_stream import IQClient, IQError, Outcome, PublishResult
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher

from iq_stream_crewai import CrewAIStream, Frame, enable_token_streaming, publish
from iq_stream_crewai._snapshot import Snapshotter

SECRET = "test-app-secret-0123456789abcdef0123"


class OrderedPublisher(RecordingPublisher):
    """Records what had happened when the close line was sent."""

    def __init__(self, saved: list[str]) -> None:
        super().__init__()
        self.saved = saved
        self.saved_at_close: list[str] | None = None

    def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        self.saved_at_close = list(self.saved)
        return super().close(outcome, timeout)


def test_the_answer_is_saved_before_the_close_line() -> None:
    saved: list[str] = []
    publisher = OrderedPublisher(saved)
    with publish(publisher):
        result = make_crew().kickoff()
        saved.append(result.raw)
    assert publisher.saved_at_close == [ANSWER]
    assert publisher.outcome == "end"


def test_a_block_with_several_kickoffs_is_one_run() -> None:
    publisher = RecordingPublisher()
    with publish(publisher):
        make_crew(name="one").kickoff()
        make_crew(name="two").kickoff()
    events = checked(publisher.data)
    assert types(events).count("RUN_STARTED") == 1
    assert types(events).count("STEP_STARTED") == 2
    assert conversation(events).text == ANSWER * 2


def test_an_exception_after_the_kickoff_fails_the_stream_and_goes_on() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(OSError, match="disk full"), publish(publisher):
        make_crew().kickoff()
        raise OSError("disk full")
    events = checked(publisher.data)
    assert events[-1] == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "OSError"}
    assert conversation(events).text == ANSWER
    assert publisher.outcome == "fail"


def test_keyboard_interrupt_is_a_cancellation() -> None:
    for on_cancel in ("fail", "end"):
        publisher = RecordingPublisher()
        with pytest.raises(KeyboardInterrupt), publish(publisher, on_cancel=on_cancel):  # type: ignore[arg-type]
            raise KeyboardInterrupt
        events = checked(publisher.data)
        assert events[-1]["outcome"] == {"type": "cancelled"}
        assert publisher.outcome == on_cancel


@pytest.mark.parametrize("on_cancel", ["fail", "end"])
async def test_a_cancelled_task_ends_with_the_cancelled_outcome(on_cancel: Outcome) -> None:
    publisher = RecordingPublisher()
    gate = threading.Event()
    started = asyncio.Event()
    loop = asyncio.get_running_loop()

    class Signalling(LookupTool):
        def _run(self, term: str = "default") -> str:
            loop.call_soon_threadsafe(started.set)
            return super()._run(term)

    async def run() -> None:
        async with publish(publisher, on_cancel=on_cancel):
            await make_crew(tool=Signalling(gate=gate, seconds=5.0)).akickoff()

    task = asyncio.create_task(run())
    await asyncio.wait_for(started.wait(), 5)  # the tool is running
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    gate.set()  # the crew's worker thread is not interrupted by the cancel; let it finish
    events = checked(publisher.data)
    assert events[-1]["type"] == "RUN_FINISHED"
    assert events[-1]["outcome"] == {"type": "cancelled"}
    assert "TOOL_CALL_START" in types(events)
    assert publisher.outcome == on_cancel


async def test_async_with_and_akickoff() -> None:
    publisher = RecordingPublisher()
    async with publish(publisher) as run:
        result = await make_crew().akickoff()
    events = checked(publisher.data)
    assert conversation(events).text == ANSWER == result.raw
    assert publisher.outcome == "end"
    assert run.result is not None and run.result.outcome == "end"


async def test_async_with_and_kickoff_async() -> None:
    publisher = RecordingPublisher()
    async with publish(publisher):
        await make_crew().kickoff_async()
    assert conversation(checked(publisher.data)).text == ANSWER


def test_an_inert_publisher_never_breaks_the_run() -> None:
    publisher = RecordingPublisher()
    publisher.fail()  # as after a fatal answer from the server
    with publish(publisher) as run:
        result = make_crew().kickoff()
    assert result.raw == ANSWER
    assert publisher.data == []
    assert run.errors == []
    assert run.result is not None and run.result.status == 0


class Dying(RecordingPublisher):
    """Gives up on the frame after ``after`` frames, as the real one does when the server refuses."""

    def __init__(self, after: int, **options: Any) -> None:
        super().__init__(**options)
        self.after = after
        self.sends = 0

    def send(self, data: str, event: str | None = None) -> bool:
        self.sends += 1
        if len(self.frames) == self.after:
            self.fail()
        return super().send(data, event)


def test_a_publisher_that_dies_mid_run_is_not_fed_and_the_run_finishes() -> None:
    publisher = Dying(after=5)
    with publish(publisher) as run:
        result = make_crew(tool=LookupTool(seconds=0.05)).kickoff()
    assert result.raw == ANSWER
    assert len(publisher.data) == 5
    assert publisher.sends == 6  # the one refused, and nothing after it: the sink reads alive
    assert run.errors == []
    assert run.result is not None and run.result.outcome is None


def test_a_refusing_publisher_raises_only_at_the_end_of_a_normal_block() -> None:
    publisher = Dying(after=3, on_error="raise")
    saved: list[str] = []
    with pytest.raises(IQError, match="the test made the publisher give up"), publish(publisher) as run:
        saved.append(make_crew().kickoff().raw)
    assert saved == [ANSWER]  # the run and its persistence completed
    assert len(publisher.data) == 3
    assert [type(error) for error in run.errors] == [IQError]
    # When the block itself fails, its own exception is the one that leaves it.
    publisher = Dying(after=3, on_error="raise")
    with pytest.raises(OSError), publish(publisher):
        make_crew().kickoff()
        raise OSError("the application's own")


def test_an_adapter_failure_is_recorded_logged_once_and_the_run_goes_on(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    original = Snapshotter.__call__

    def broken(self: Snapshotter, source: object, event: BaseEvent) -> Any:
        if event.type in ("tool_usage_started", "llm_stream_chunk"):
            raise ValueError(f"broken snapshot of {ANSWER}")
        return original(self, source, event)

    monkeypatch.setattr(Snapshotter, "__call__", broken)
    publisher = RecordingPublisher()
    with caplog.at_level(logging.WARNING, logger="iq_stream_crewai"), publish(publisher) as run:
        result = make_crew().kickoff()
    assert result.raw == ANSWER
    assert len(run.errors) == 13 and all(isinstance(error, ValueError) for error in run.errors)
    [record] = [record for record in caplog.records if record.name == "iq_stream_crewai"]
    assert "ValueError" in record.getMessage()
    assert ANSWER not in record.getMessage()  # no event data in a log line
    # The text came as one block when the call returned; the tool result found no started call to attach to and was
    # announced with it.
    events = checked(publisher.data)
    assert conversation(events).text == ANSWER
    assert publisher.outcome == "end"


def test_an_encoder_failure_mid_event_does_not_break_the_run(monkeypatch: pytest.MonkeyPatch) -> None:
    from iq_stream_crewai._agui import AgUiEncoder

    def broken(self: AgUiEncoder, snapshot: Any) -> list[Frame]:
        raise RuntimeError("encoder bug")

    monkeypatch.setattr(AgUiEncoder, "_tool_ended", broken)
    publisher = RecordingPublisher()
    with publish(publisher) as run:
        result = make_crew().kickoff()
    assert result.raw == ANSWER
    assert [str(error) for error in run.errors] == ["encoder bug"]
    assert "TOOL_CALL_RESULT" not in types(checked(publisher.data))


def test_a_thread_that_outlives_the_block_publishes_nothing() -> None:
    publisher = RecordingPublisher()
    late = threading.Event()
    failures: list[BaseException] = []

    def straggler() -> None:
        late.wait(5)
        try:
            crewai_event_bus.emit(None, CrewKickoffStartedEvent(crew_name="late", inputs=None))
        except BaseException as error:
            failures.append(error)

    with publish(publisher, protocol="crewai") as run:
        thread = threading.Thread(target=contextvars.copy_context().run, args=(straggler,))
        thread.start()
    count = len(publisher.frames)
    late.set()
    thread.join(5)
    assert len(publisher.frames) == count == 0
    assert failures == [] and run.errors == []


def test_the_application_s_own_frames_go_in_order() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai") as run:
        assert run.send('{"note":"before"}', event="app")
        make_crew(terms=[]).kickoff()
        assert run.send('{"note":"after"}', event="app")
    assert publisher.frames[0].event == "app" and publisher.frames[-1].event == "app"
    assert run.frames_sent == len(publisher.frames)
    with pytest.raises(RuntimeError, match="inside the block"):
        run.send("late")


def test_a_custom_encoder_and_a_frame_that_does_not_fit() -> None:
    class Names:
        def snapshot(self, source: object, event: BaseEvent) -> str | None:
            return event.type if event.type.startswith("task_") else None

        def start(self) -> Iterable[Frame]:
            return [("start", "{}")]

        def frames(self, snapshot: str) -> Iterable[Frame]:
            yield ("task", json.dumps({"type": snapshot}))
            yield (None, "x" * 2000)  # over the limit: dropped and recorded

        def finish(self, error: BaseException | None) -> Iterable[Frame]:
            return [("finish", json.dumps({"error": type(error).__name__ if error else None}))]

    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publish(publisher, protocol=Names()) as run:
        make_crew().kickoff()
    assert [frame.event for frame in publisher.frames] == ["start", "task", "task", "finish"]
    assert [type(error).__name__ for error in run.errors] == ["FrameTooLarge", "FrameTooLarge"]


def test_an_iq_client_publisher_is_created_when_the_block_is_entered() -> None:
    created: list[RecordingPublisher] = []

    class Client(IQClient):
        def publisher(self, stream_id: str) -> RecordingPublisher:  # type: ignore[override]
            publisher = RecordingPublisher(stream_id)
            created.append(publisher)
            return publisher

    iq = Client("http://127.0.0.1:9", "demo-app", SECRET)
    run = publish(iq)
    assert created == []  # no I/O before the block
    assert len(run.stream_id) == 22 and run.thread_id == run.stream_id
    with run:
        make_crew(terms=[]).kickoff()
    [publisher] = created
    assert publisher.stream_id == run.stream_id
    assert publisher.outcome == "end"
    assert run.publisher is publisher
    with pytest.raises(RuntimeError, match="entered once"), run:
        pass


def test_arguments_are_checked_before_anything_is_sent() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(ValueError, match="stream id"):
        publish(IQClient("http://127.0.0.1:9", "demo-app", SECRET), "short")
    with pytest.raises(ValueError, match="differs"):
        publish(publisher, "x" * 22)
    with pytest.raises(ValueError, match="protocol"):
        publish(publisher, protocol="ai-sdk")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="on_cancel"):
        publish(publisher, on_cancel="abort")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="transport"):
        publish(publisher, transport="queue")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="max_text_chars"):
        publish(publisher, max_text_chars=-1)
    with pytest.raises(TypeError, match="synchronous publisher"):
        publish(AsyncRecordingPublisher())  # type: ignore[arg-type]
    assert publisher.data == []
    assert isinstance(publish(publisher), CrewAIStream)


def test_enable_token_streaming() -> None:
    crew = make_crew(stream=False)
    [agent] = crew.agents
    assert isinstance(agent.llm, FakeStreamingLLM) and not agent.llm.stream
    enable_token_streaming(crew)
    assert agent.llm.stream
    llm = FakeStreamingLLM(stream=False)
    enable_token_streaming(Agent(role="r", goal="g", backstory="b", llm=llm))
    assert llm.stream
    other = FakeStreamingLLM(stream=False)
    enable_token_streaming(other)
    assert other.stream
    with pytest.raises(TypeError, match="crews, agents and models"):
        enable_token_streaming("gpt-5")  # type: ignore[arg-type]
    publisher = RecordingPublisher()
    with publish(publisher):
        crew.kickoff()
    assert types(checked(publisher.data)).count("TEXT_MESSAGE_CONTENT") == 6
