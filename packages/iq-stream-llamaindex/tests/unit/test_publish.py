"""publish() and published(): the close line for every way a run ends, persistence before the close, failing open,
the tee, and the errors that belong to the application."""

from __future__ import annotations

import asyncio
import gc
import json
import logging
import time
import warnings
from collections.abc import AsyncGenerator, Awaitable, Callable, Iterable
from typing import Any

import pytest
from fakes import (
    ANSWER,
    FIRST_TEXT,
    REACT_ANSWER,
    ScriptedLLM,
    broken_weather,
    get_weather,
    hitl_agent,
    needs_agent_run_id,
    needs_agent_waits,
    react_agent,
    slow_weather,
    tool_call,
    weather_agent,
)
from iq_stream import IQError, Outcome, PublishResult
from iq_stream.testing import AsyncRecordingPublisher
from llama_index.core.agent.workflow import FunctionAgent, ToolCall
from llama_index.core.base.llms.types import TextBlock
from workflows import Context, Workflow, step
from workflows.errors import WorkflowCancelledByUser, WorkflowTimeoutError
from workflows.events import (
    Event,
    HumanResponseEvent,
    InputRequiredEvent,
    StartEvent,
    StopEvent,
    WorkflowCancelledEvent,
    WorkflowFailedEvent,
    WorkflowTimedOutEvent,
)

from iq_stream_llamaindex import AGUIEncoder, Frame, NativeEncoder, RunOutcome, publish, published
from iq_stream_llamaindex import _publish as publish_module


def names(publisher: AsyncRecordingPublisher) -> list[str | None]:
    return [frame.event for frame in publisher.frames]


def agui_types(publisher: AsyncRecordingPublisher) -> list[str]:
    return [json.loads(data)["type"] for data in publisher.data]


async def wait_for(condition: object, seconds: float = 5.0) -> None:
    deadline = time.monotonic() + seconds
    while not condition():  # type: ignore[operator]
        assert time.monotonic() < deadline, "timed out"
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------------------------------- success


async def test_a_completed_run_publishes_every_frame_in_order_then_closes_with_end() -> None:
    publisher = AsyncRecordingPublisher()
    outcome = await publish(weather_agent().run(user_msg="Weather in Paris?"), publisher)
    expected = (
        ["RunStarted", "AgentStream"]  # the reasoning delta
        + ["AgentStream"] * 5
        + ["AgentOutput", "ToolCall", "ToolCallResult"]
        + ["AgentStream"] * 7
        + ["AgentOutput", "StopEvent"]
    )
    assert names(publisher) == expected
    assert publisher.outcome == "end" and publisher.close_calls == 1
    assert outcome.status == "completed" and outcome.error is None
    assert str(outcome.result) == ANSWER
    assert isinstance(outcome.stop_event, StopEvent)
    assert outcome.frames == len(publisher.frames) and outcome.published
    assert outcome.publish_result == PublishResult(
        status=200, last_event_id=len(publisher.frames), closed=True, outcome="end"
    )
    assert json.loads(publisher.data[0]) == {"run_id": outcome.run_id}


@needs_agent_run_id
async def test_the_run_id_is_the_stream_id_when_the_application_passes_it() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent().run(user_msg="Weather?", run_id=publisher.stream_id)
    outcome = await publish(handler, publisher, encoder=AGUIEncoder(thread_id="thread-1"))
    started = json.loads(publisher.data[0])
    assert started == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": publisher.stream_id}
    assert outcome.run_id == publisher.stream_id


# ---------------------------------------------------------------------------------------------------- persistence


async def test_before_close_runs_after_the_last_event_and_before_the_terminal_frame_and_the_close() -> None:
    publisher = AsyncRecordingPublisher()
    seen: dict[str, object] = {}

    async def before_close(outcome: RunOutcome) -> None:
        await asyncio.sleep(0)
        seen.update(outcome=publisher.outcome, last=publisher.frames[-1].event, status=outcome.status)
        seen["result"] = str(outcome.result)

    await publish(weather_agent().run(user_msg="Weather?"), publisher, before_close=before_close)
    assert seen == {"outcome": None, "last": "AgentOutput", "status": "completed", "result": ANSWER}
    assert names(publisher)[-1] == "StopEvent" and publisher.outcome == "end"


async def test_a_synchronous_before_close_is_called_too() -> None:
    calls: list[str] = []
    publisher = AsyncRecordingPublisher()
    await publish(weather_agent().run(user_msg="Weather?"), publisher, before_close=lambda o: calls.append(o.status))
    assert calls == ["completed"]


@pytest.mark.parametrize("encoder", [NativeEncoder(), AGUIEncoder(thread_id="t")], ids=["native", "agui"])
async def test_before_close_raising_aborts_the_stream_with_fail_and_propagates(encoder: NativeEncoder) -> None:
    publisher = AsyncRecordingPublisher()

    def before_close(outcome: RunOutcome) -> None:
        raise OSError("the database is down")

    with pytest.raises(OSError, match="database"):
        await publish(weather_agent().run(user_msg="Weather?"), publisher, encoder=encoder, before_close=before_close)
    assert publisher.outcome == "fail"
    last = json.loads(publisher.data[-1])
    if isinstance(encoder, NativeEncoder):
        assert names(publisher)[-1] == "RunAborted" and last == {"reason": "error", "message": "An error occurred."}
        assert "StopEvent" not in names(publisher)
    else:
        assert last == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "aborted"}
        assert "RUN_FINISHED" not in agui_types(publisher)


# ---------------------------------------------------------------------------------------------------- failure


@pytest.mark.parametrize("encoder", [NativeEncoder(), AGUIEncoder(thread_id="t")], ids=["native", "agui"])
async def test_a_failed_run_closes_with_fail_after_its_error_frame_and_is_returned(encoder: NativeEncoder) -> None:
    publisher = AsyncRecordingPublisher()
    statuses: list[str] = []
    outcome = await publish(
        weather_agent(fail_on_second_call=True).run(user_msg="Weather?"),
        publisher,
        encoder=encoder,
        before_close=lambda o: statuses.append(o.status),
    )
    assert statuses == ["failed"] and publisher.outcome == "fail"
    assert outcome.status == "failed" and isinstance(outcome.error, RuntimeError)
    assert isinstance(outcome.stop_event, WorkflowFailedEvent) and outcome.result is None
    last = json.loads(publisher.data[-1])
    if isinstance(encoder, NativeEncoder):
        assert names(publisher)[-1] == "WorkflowFailedEvent"
        assert last == {"step_name": "run_agent_step", "message": "An error occurred."}
    else:
        assert last == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "failed"}
    assert not any("sk-hidden" in data for data in publisher.data)


async def test_raise_run_errors_raises_the_models_exception_after_the_close() -> None:
    publisher = AsyncRecordingPublisher()
    with pytest.raises(RuntimeError, match="exploded"):
        await publish(
            weather_agent(fail_on_second_call=True).run(user_msg="Weather?"), publisher, raise_run_errors=True
        )
    assert publisher.outcome == "fail"


async def test_a_tool_that_raises_does_not_fail_the_run_and_its_error_text_is_masked() -> None:
    publisher = AsyncRecordingPublisher()
    outcome = await publish(weather_agent(tool=broken_weather).run(user_msg="Weather?"), publisher)
    assert outcome.status == "completed" and publisher.outcome == "end"
    result = json.loads(publisher.data[names(publisher).index("ToolCallResult")])
    assert result["is_error"] is True and result["content"] == "An error occurred."
    assert not any("sk-tool-secret" in data for data in publisher.data)


async def test_a_timed_out_run_closes_with_fail() -> None:
    publisher = AsyncRecordingPublisher()
    agent = weather_agent(tool=slow_weather(5.0), timeout=0.3)
    outcome = await publish(agent.run(user_msg="Weather?"), publisher, encoder=AGUIEncoder(thread_id="t"))
    assert outcome.status == "timed_out" and isinstance(outcome.error, WorkflowTimeoutError)
    assert isinstance(outcome.stop_event, WorkflowTimedOutEvent)
    assert json.loads(publisher.data[-1]) == {"type": "RUN_ERROR", "message": "The run timed out.", "code": "timeout"}
    assert publisher.outcome == "fail"


# ---------------------------------------------------------------------------------------------------- cancellation


@pytest.mark.parametrize("close_on_cancel", ["fail", "end"])
async def test_cancel_run_during_a_tool_closes_with_the_chosen_line_after_the_cancelled_outcome(
    close_on_cancel: Outcome,
) -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            asyncio.get_running_loop().create_task(handler.cancel_run())

    outcome = await publish(
        handler, publisher, encoder=AGUIEncoder(thread_id="t"), on_event=on_event, close_on_cancel=close_on_cancel
    )
    assert outcome.status == "cancelled" and isinstance(outcome.error, WorkflowCancelledByUser)
    assert isinstance(outcome.stop_event, WorkflowCancelledEvent)
    assert agui_types(publisher)[-2:] == ["STEP_FINISHED", "RUN_FINISHED"]
    assert json.loads(publisher.data[-1])["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == close_on_cancel


async def test_cancelling_the_publishing_task_stops_the_run_aborts_and_closes_with_fail() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    task = asyncio.create_task(publish(handler, publisher, encoder=AGUIEncoder(thread_id="t")))
    await wait_for(lambda: "TOOL_CALL_END" in agui_types(publisher))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handler.is_done()
    with pytest.raises(WorkflowCancelledByUser):
        await handler
    assert json.loads(publisher.data[-1])["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "fail"


async def test_cancelling_the_task_with_the_native_encoder_publishes_run_aborted() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    task = asyncio.create_task(publish(handler, publisher))
    await wait_for(lambda: "ToolCall" in names(publisher))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert names(publisher)[-1] == "RunAborted" and json.loads(publisher.data[-1]) == {"reason": "cancelled"}
    assert publisher.outcome == "fail"


async def test_a_hard_cancelled_run_ends_within_the_grace_period_instead_of_hanging() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                handler.cancel()  # the deprecated hard cancel: no terminal event is ever published

    started = time.monotonic()
    outcome = await asyncio.wait_for(
        publish(handler, publisher, encoder=AGUIEncoder(thread_id="t"), on_event=on_event), 10
    )
    assert time.monotonic() - started < publish_module.GRACE + 3
    assert outcome.status == "cancelled" and isinstance(outcome.error, WorkflowCancelledByUser)
    assert json.loads(publisher.data[-1])["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "fail"


# ---------------------------------------------------------------------------------------------------- human in the loop


class Approval(Workflow):
    @step
    async def ask(self, ctx: Context, ev: StartEvent) -> StopEvent:
        waiter = InputRequiredEvent(prefix="Proceed?")
        answer = await ctx.wait_for_event(HumanResponseEvent, waiter_id="ok", waiter_event=waiter, timeout=10)
        return StopEvent(result=answer.response)


async def test_a_workflow_waiting_for_a_human_keeps_the_stream_open_until_the_answer() -> None:
    publisher = AsyncRecordingPublisher()
    handler = Approval(timeout=10).run()
    paused: list[tuple[bool, Outcome | None]] = []

    async def on_event(event: Event) -> None:
        if isinstance(event, InputRequiredEvent):
            await asyncio.sleep(0.2)
            paused.append((handler.is_done(), publisher.outcome))
            handler.ctx.send_event(HumanResponseEvent(response="yes"))

    outcome = await publish(handler, publisher, on_event=on_event)
    assert paused == [(False, None)] and outcome.result == "yes" and publisher.outcome == "end"
    assert names(publisher) == ["RunStarted", "InputRequiredEvent", "StopEvent"]


@needs_agent_waits
@pytest.mark.parametrize("encoder", [NativeEncoder(), AGUIEncoder(thread_id="t")], ids=["native", "agui"])
async def test_a_human_input_pause_keeps_the_stream_open_until_the_answer_and_the_run_ends(
    encoder: NativeEncoder,
) -> None:
    publisher = AsyncRecordingPublisher()
    handler = hitl_agent().run(user_msg="Do it")
    paused: list[tuple[bool, Outcome | None]] = []

    async def on_event(event: Event) -> None:
        if isinstance(event, InputRequiredEvent):
            await asyncio.sleep(0.2)  # the browser takes its time; nothing is closed meanwhile
            paused.append((handler.is_done(), publisher.outcome))
            handler.ctx.send_event(HumanResponseEvent(response="yes"))

    outcome = await publish(handler, publisher, encoder=encoder, on_event=on_event)
    assert paused == [(False, None)]
    assert outcome.status == "completed" and publisher.outcome == "end" and str(outcome.result) == "Done."
    if isinstance(encoder, NativeEncoder):
        assert names(publisher).count("ToolCall") == 1  # the replayed tool step is announced once
        pause = json.loads(publisher.data[names(publisher).index("InputRequiredEvent")])
        assert pause == {"prefix": "Proceed?"}
    else:
        kinds = agui_types(publisher)
        assert kinds.count("TOOL_CALL_START") == 1 and kinds.count("TOOL_CALL_RESULT") == 1
        custom = json.loads(publisher.data[kinds.index("CUSTOM")])
        assert custom["name"] == "input_required" and custom["value"] == {"prefix": "Proceed?"}
        result = json.loads(publisher.data[kinds.index("TOOL_CALL_RESULT")])
        assert result["content"] == "yes"


# ---------------------------------------------------------------------------------------------------- failing open


async def test_an_inert_publisher_never_breaks_the_run_which_is_read_to_its_end() -> None:
    publisher = AsyncRecordingPublisher()
    publisher.fail()  # as after a refusal from the server
    saved: list[RunOutcome] = []
    handler = weather_agent().run(user_msg="Weather?")
    outcome = await publish(handler, publisher, before_close=saved.append)
    assert outcome.status == "completed" and str(outcome.result) == ANSWER and handler.is_done()
    assert saved and not saved[0].published and not outcome.published
    assert publisher.frames == [] and publisher.close_calls == 0 and outcome.publish_result is None


class GivesUpAt(AsyncRecordingPublisher):
    """Gives up when asked to send its ``at``-th frame, as a publisher does after a fatal answer."""

    def __init__(self, at: int) -> None:
        super().__init__()
        self.at = at
        self.calls = 0

    def send(self, data: str, event: str | None = None) -> bool:
        self.calls += 1
        if self.calls == self.at:
            self.fail()
        return super().send(data, event)


async def test_a_publisher_that_gives_up_mid_run_stops_the_encoding_and_the_run_completes() -> None:
    publisher = GivesUpAt(3)
    saved: list[RunOutcome] = []
    outcome = await publish(weather_agent().run(user_msg="Weather?"), publisher, before_close=saved.append)
    assert len(publisher.frames) == 2 and publisher.calls == 3  # nothing was encoded or sent after the refusal
    assert outcome.status == "completed" and str(outcome.result) == ANSWER
    assert saved[0].status == "completed" and not outcome.published and outcome.frames == 2
    assert publisher.close_calls == 0


async def test_with_on_error_raise_the_publishers_error_stops_the_run_and_propagates() -> None:
    publisher = AsyncRecordingPublisher(on_error="raise")
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            publisher.fail()

    with pytest.raises(IQError):
        await publish(handler, publisher, on_event=on_event)
    assert handler.is_done()
    with pytest.raises(WorkflowCancelledByUser):
        await handler


class Breaks(NativeEncoder):
    """Raises on the tool result, as an encoder may on a value it cannot handle."""

    def encode(self, event: Event) -> Iterable[Frame]:
        if type(event).__name__ == "ToolCallResult":
            raise TypeError("cannot encode this")
        return super().encode(event)


async def test_an_encoder_failing_on_one_event_skips_its_frames_and_the_stream_goes_on(
    caplog: pytest.LogCaptureFixture,
) -> None:
    publisher = AsyncRecordingPublisher()
    with caplog.at_level(logging.WARNING, logger="iq_stream_llamaindex"):
        outcome = await publish(weather_agent().run(user_msg="Weather?"), publisher, encoder=Breaks())
    assert "ToolCallResult" not in names(publisher) and names(publisher)[-1] == "StopEvent"
    assert outcome.status == "completed" and publisher.outcome == "end"
    assert "the encoder failed on ToolCallResult (TypeError)" in caplog.text
    assert "cannot encode this" not in caplog.text


# ---------------------------------------------------------------------------------------------------- the tee


async def test_published_yields_each_event_after_its_frames_and_the_outcome_after_the_block() -> None:
    publisher = AsyncRecordingPublisher()
    seen: list[tuple[str, int]] = []
    async with published(weather_agent().run(user_msg="Weather?"), publisher) as events:
        async for event in events:
            seen.append((type(event).__name__, len(publisher.frames)))
    tool_call_frame = names(publisher).index("ToolCall") + 1
    assert ("ToolCall", tool_call_frame) in seen
    assert seen[-1][0] == "StopEvent" and publisher.outcome == "end"
    assert events.outcome.status == "completed"


async def test_breaking_out_of_the_tee_still_publishes_the_rest_of_the_run_and_closes_with_end() -> None:
    publisher = AsyncRecordingPublisher()
    async with published(weather_agent().run(user_msg="Weather?"), publisher) as events:
        async for event in events:
            if isinstance(event, ToolCall):
                break
    assert names(publisher)[-1] == "StopEvent" and publisher.outcome == "end"
    streamed = [frame.data for frame in publisher.frames if frame.event == "AgentStream"]
    answer = "".join(json.loads(data)["delta"] for data in streamed)
    assert answer == FIRST_TEXT + ANSWER
    assert events.outcome.status == "completed"


async def test_an_exception_in_the_tee_cancels_the_run_closes_with_fail_and_propagates() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    with pytest.raises(LookupError):
        # close_on_cancel is for a stop: an application that failed closes with fail whatever it says
        async with published(handler, publisher, encoder=AGUIEncoder(thread_id="t"), close_on_cancel="end") as events:
            async for event in events:
                if isinstance(event, ToolCall):
                    raise LookupError("the application gave up")
    assert handler.is_done() and publisher.outcome == "fail"
    assert json.loads(publisher.data[-1]) == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "aborted"}
    assert events.outcome.status == "cancelled" and isinstance(events.outcome.error, WorkflowCancelledByUser)


async def test_an_exception_from_on_event_propagates_and_closes_with_fail() -> None:
    publisher = AsyncRecordingPublisher()

    def on_event(event: Event) -> None:
        if isinstance(event, ToolCall):
            raise KeyError("application bug")

    with pytest.raises(KeyError):
        await publish(weather_agent().run(user_msg="Weather?"), publisher, on_event=on_event)
    assert publisher.outcome == "fail"


async def test_the_tee_is_used_inside_its_block_and_its_outcome_after_it() -> None:
    stream = published(weather_agent().run(user_msg="Weather?"), AsyncRecordingPublisher())
    with pytest.raises(RuntimeError, match="async with"):
        await anext(stream)
    with pytest.raises(RuntimeError, match="after the async with block"):
        _ = stream.outcome
    async with stream:
        pass
    assert stream.outcome.status == "completed"
    with pytest.raises(RuntimeError, match="entered once"):
        async with stream:
            pass


async def test_close_on_cancel_is_end_or_fail() -> None:
    handler = weather_agent().run(user_msg="Weather?")
    with pytest.raises(ValueError, match="close_on_cancel"):
        published(handler, AsyncRecordingPublisher(), close_on_cancel="maybe")  # type: ignore[arg-type]
    await publish(handler, AsyncRecordingPublisher())


async def test_a_handler_whose_events_were_already_read_is_refused_clearly_not_waited_on() -> None:
    handler = weather_agent().run(user_msg="Weather?")
    async for _ in handler.stream_events():
        pass
    publisher = AsyncRecordingPublisher()
    with pytest.raises(RuntimeError, match="single consumer"):
        await asyncio.wait_for(publish(handler, publisher), 5)
    assert publisher.outcome == "fail"


# ---------------------------------------------------------------------------------------------------- plain workflows


class Progress(Event):
    step: int


class Report(Workflow):
    @step
    async def run_report(self, ctx: Context, ev: StartEvent) -> StopEvent:
        ctx.write_event_to_stream(Progress(step=1, note="halfway"))
        return StopEvent(result={"pages": 3, "title": "Q3"})


async def test_a_plain_workflow_publishes_its_own_events_and_its_result() -> None:
    publisher = AsyncRecordingPublisher()
    outcome = await publish(Report(timeout=10).run(), publisher)
    assert names(publisher) == ["RunStarted", "Progress", "StopEvent"]
    assert json.loads(publisher.data[1]) == {"step": 1, "note": "halfway"}
    assert json.loads(publisher.data[2]) == {"result": {"pages": 3, "title": "Q3"}}
    assert outcome.result == {"pages": 3, "title": "Q3"} and publisher.outcome == "end"


# ---------------------------------------------------------------------------------------------------- concurrency


async def test_two_concurrent_runs_publish_to_their_own_streams_without_mixing() -> None:
    first, second = AsyncRecordingPublisher(), AsyncRecordingPublisher()
    agent = weather_agent()
    one = agent.run(user_msg="Weather?")
    two = agent.run(user_msg="Weather?")
    await asyncio.gather(
        publish(one, first, encoder=AGUIEncoder(thread_id="one")),
        publish(two, second, encoder=AGUIEncoder(thread_id="two")),
    )
    assert one.run_id != two.run_id
    for publisher, run_id in ((first, one.run_id), (second, two.run_id)):
        events = [json.loads(data) for data in publisher.data]
        assert {event.get("runId") for event in events} - {None} == {run_id}
        text = "".join(event["delta"] for event in events if event["type"] == "TEXT_MESSAGE_CONTENT")
        assert text == FIRST_TEXT + ANSWER and publisher.outcome == "end"


async def test_a_handler_whose_stream_another_generator_holds_is_refused_and_never_closed_with_end() -> None:
    handler = weather_agent().run(user_msg="Weather?")
    held = handler.stream_events()
    async for _ in held:
        break  # break leaves the generator suspended: it keeps the run's stream
    publisher = AsyncRecordingPublisher()
    try:
        with pytest.raises(RuntimeError, match="single consumer"):
            await asyncio.wait_for(publish(handler, publisher, encoder=AGUIEncoder(thread_id="t")), 10)
    finally:
        await held.aclose()
    assert publisher.outcome == "fail"
    assert agui_types(publisher) == ["RUN_STARTED", "RUN_ERROR"]


async def test_a_run_whose_stream_never_delivers_its_terminal_event_closes_with_fail() -> None:
    handler = Report(timeout=10).run()

    async def without_terminal() -> AsyncGenerator[Event, None]:
        yield Progress(step=1)
        await asyncio.Event().wait()  # the terminal event never comes, though the run ends

    handler.stream_events = without_terminal  # type: ignore[method-assign]
    publisher = AsyncRecordingPublisher()
    outcome = await asyncio.wait_for(publish(handler, publisher), 10)
    assert names(publisher) == ["RunStarted", "Progress", "RunAborted"]
    assert publisher.outcome == "fail" and outcome.result == {"pages": 3, "title": "Q3"}


@pytest.mark.parametrize("delay", [0, 0.05], ids=["reader-first", "reader-late"])
@pytest.mark.parametrize("protocol", ["native", "agui"])
async def test_an_agent_output_is_published_as_llama_index_finalizes_it_however_fast_it_is_read(
    protocol: str, delay: float
) -> None:
    publisher = AsyncRecordingPublisher()

    async def on_event(event: Event) -> None:
        await asyncio.sleep(delay)  # an application that awaits lets llama-index-core finalize the output first

    encoder = AGUIEncoder(thread_id="t") if protocol == "agui" else NativeEncoder()
    outcome = await publish(react_agent().run(user_msg="Weather?"), publisher, encoder=encoder, on_event=on_event)
    assert str(outcome.result) == REACT_ANSWER
    if protocol == "agui":
        texts = [json.loads(data)["delta"] for data in publisher.data if '"TEXT_MESSAGE_CONTENT"' in data]
        assert texts[-1] == REACT_ANSWER
    else:
        final = json.loads(publisher.data[names(publisher).index("StopEvent") - 1])
        assert names(publisher)[-2] == "AgentOutput" and final["content"] == REACT_ANSWER
        assert [call["tool_name"] for call in final["tool_calls"]] == ["get_weather"]  # the run's, as in the result


async def test_a_function_agents_last_output_lists_the_runs_tool_calls_however_fast_it_is_read() -> None:
    for delay in (0, 0.05):
        publisher = AsyncRecordingPublisher()

        async def on_event(event: Event, delay: float = delay) -> None:
            await asyncio.sleep(delay)

        outcome = await publish(weather_agent().run(user_msg="Weather?"), publisher, on_event=on_event)
        final = json.loads(publisher.data[-2])
        assert final["tool_calls"] == [
            {"tool_id": "call_1", "tool_name": "get_weather", "tool_kwargs": {"city": "Paris"}}
        ]
        assert len(outcome.result.tool_calls) == 1


async def test_the_tee_yields_the_terminal_event_before_its_frames_and_the_close() -> None:
    publisher = AsyncRecordingPublisher()
    at_terminal: list[tuple[str | None, Outcome | None]] = []
    async with published(weather_agent().run(user_msg="Weather?"), publisher) as events:
        async for event in events:
            if isinstance(event, StopEvent):
                at_terminal.append((publisher.frames[-1].event, publisher.outcome))
    assert at_terminal == [("AgentOutput", None)]  # the loop can still save the answer before the browser learns
    assert names(publisher)[-1] == "StopEvent" and publisher.outcome == "end"


async def test_a_second_cancellation_of_the_publishing_task_still_stops_the_run_and_closes_with_fail() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    task = asyncio.create_task(publish(handler, publisher))
    await wait_for(lambda: "ToolCall" in names(publisher))
    task.cancel()
    await asyncio.sleep(0)  # the first cancellation is delivered and the cleanup starts
    task.cancel()  # a server shutting down cancels again
    with pytest.raises(asyncio.CancelledError):
        await task
    await wait_for(lambda: publisher.outcome is not None, 8)
    assert publisher.outcome == "fail" and names(publisher)[-1] == "RunAborted" and handler.is_done()


async def test_a_run_whose_result_does_not_settle_after_its_stream_ended_fails_without_hanging(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(publish_module, "SETTLE", 0.3)
    handler = Report(timeout=10).run()
    release = asyncio.Event()

    async def stuck() -> StopEvent:
        await release.wait()
        raise RuntimeError("settled late")

    handler.stop_event_result = stuck  # type: ignore[method-assign]
    publisher = AsyncRecordingPublisher()
    asyncio.get_running_loop().call_later(3, release.set)  # bounds the test if publish() waits for ever
    started = time.monotonic()
    with caplog.at_level(logging.ERROR, logger="asyncio"):
        outcome = await publish(handler, publisher)
        assert time.monotonic() - started < 2
        assert outcome.status == "failed" and "did not end" in str(outcome.error) and publisher.outcome == "fail"
        release.set()
        await asyncio.sleep(0.05)
        del outcome
        gc.collect()
    assert "never retrieved" not in caplog.text  # the late exception is marked as seen


# ---------------------------------------------------------------------------------------------------- round two


async def test_cancelling_the_publishing_task_closes_with_close_on_cancel() -> None:
    publisher = AsyncRecordingPublisher()
    handler = weather_agent(tool=slow_weather(5.0)).run(user_msg="Weather?")
    task = asyncio.create_task(publish(handler, publisher, encoder=AGUIEncoder(thread_id="t"), close_on_cancel="end"))
    await wait_for(lambda: "TOOL_CALL_END" in agui_types(publisher))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert handler.is_done() and json.loads(publisher.data[-1])["outcome"] == {"type": "cancelled"}
    assert publisher.outcome == "end"


def answering_agent(structured: Callable[[Any], Awaitable[dict[str, Any]]]) -> FunctionAgent:
    """An agent built with streaming=False, whose answer is in its last AgentOutput, with a structured output that
    llama-index-core computes after it has written that output to the stream."""
    llm = ScriptedLLM(script=[[tool_call("get_weather", "c1", city="Paris")], [TextBlock(text=ANSWER)]])
    return FunctionAgent(name="a", tools=[get_weather], llm=llm, streaming=False, structured_output_fn=structured)


@pytest.mark.parametrize("protocol", ["native", "agui"])
async def test_the_last_output_waits_a_bounded_time_for_its_finalization_not_for_a_structured_output_call(
    protocol: str,
) -> None:
    marks: dict[str, float] = {}

    async def structured(messages: Any) -> dict[str, Any]:
        marks["written"] = time.monotonic()  # the answer is in the stream already
        await asyncio.sleep(1.5)  # a second model call
        return {"city": "Paris"}

    class Timed(AsyncRecordingPublisher):
        def send(self, data: str, event: str | None = None) -> bool:
            if ANSWER in data:
                marks.setdefault("sent", time.monotonic())
            return super().send(data, event)

    publisher = Timed()
    encoder = AGUIEncoder(thread_id="t") if protocol == "agui" else NativeEncoder()
    outcome = await publish(answering_agent(structured).run(user_msg="Weather?"), publisher, encoder=encoder)
    assert outcome.status == "completed" and publisher.outcome == "end"
    assert marks["sent"] - marks["written"] < publish_module.HOLD + 0.5


@pytest.mark.parametrize("protocol", ["native", "agui"])
async def test_an_answer_held_for_its_finalization_is_published_when_the_task_is_cancelled(
    protocol: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(publish_module, "HOLD", 30.0)
    computing = asyncio.Event()

    async def structured(messages: Any) -> dict[str, Any]:
        computing.set()
        await asyncio.sleep(30)
        return {}

    publisher = AsyncRecordingPublisher()
    encoder = AGUIEncoder(thread_id="t") if protocol == "agui" else NativeEncoder()
    task = asyncio.create_task(publish(answering_agent(structured).run(user_msg="?"), publisher, encoder=encoder))
    await asyncio.wait_for(computing.wait(), 5)
    await asyncio.sleep(0.2)  # the output is read and held
    task.cancel()  # the user pressed stop
    with pytest.raises(asyncio.CancelledError):
        await task
    assert publisher.outcome == "fail"
    if protocol == "native":
        assert names(publisher)[-2:] == ["AgentOutput", "RunAborted"]
        assert json.loads(publisher.data[-2])["content"] == ANSWER
    else:
        kinds = agui_types(publisher)
        assert json.loads(publisher.data[kinds.index("TEXT_MESSAGE_CONTENT")])["delta"] == ANSWER
        assert kinds[-1] == "RUN_FINISHED" and json.loads(publisher.data[-1])["outcome"] == {"type": "cancelled"}


async def test_one_encoder_shared_by_concurrent_runs_keeps_each_runs_frames_its_own() -> None:
    for encoder in (NativeEncoder(), AGUIEncoder(thread_id="t")):  # configured once, as in a route module
        publishers = [AsyncRecordingPublisher(), AsyncRecordingPublisher()]
        handlers = [weather_agent(tool=slow_weather(0.3)).run(user_msg="Weather?") for _ in publishers]
        await asyncio.gather(*(publish(h, p, encoder=encoder) for h, p in zip(handlers, publishers, strict=True)))
        for handler, publisher in zip(handlers, publishers, strict=True):
            assert publisher.outcome == "end"
            if isinstance(encoder, NativeEncoder):
                (call,) = [json.loads(frame.data) for frame in publisher.frames if frame.event == "ToolCall"]
                assert call["call_id"] == "call_1" and json.loads(publisher.data[0])["run_id"] == handler.run_id
            else:
                events = [json.loads(data) for data in publisher.data]
                assert {event["runId"] for event in events if "runId" in event} == {handler.run_id}
                assert [event["toolCallId"] for event in events if event["type"] == "TOOL_CALL_START"] == ["call_1"]


class Slow(Workflow):
    @step
    async def wait(self, ctx: Context, ev: StartEvent) -> StopEvent:
        await asyncio.sleep(5)
        return StopEvent(result="late")


async def test_a_hard_cancel_before_any_event_is_reported_as_cancelled_not_as_a_consumed_stream() -> None:
    publisher = AsyncRecordingPublisher()
    handler = Slow(timeout=10).run()

    def hard_cancel() -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            handler.cancel()

    asyncio.get_running_loop().call_later(0.1, hard_cancel)
    outcome = await asyncio.wait_for(publish(handler, publisher), 10)
    assert outcome.status == "cancelled" and isinstance(outcome.error, WorkflowCancelledByUser)
    assert names(publisher) == ["RunStarted", "RunAborted"] and publisher.outcome == "fail"
