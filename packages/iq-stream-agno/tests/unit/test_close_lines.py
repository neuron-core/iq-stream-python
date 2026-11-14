"""Which close line, decided from the root run: completion, failure, cancellation, a pause, a caller that stops, a
cancelled task; teams and workflows; and the run saved before the close line goes out."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from typing import Any

import pytest
from agno.agent import Agent
from agno.db.in_memory import InMemoryDb
from agno.team import Team
from agno.workflow import Step, Workflow
from iq_stream import Outcome, PublishResult
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from recorded import collect, names, payloads
from scripted_model import SLOW, ScriptedModel, get_weather, weather_agent

from iq_stream_agno import RunSummary, apublish, publish


def test_a_caller_that_stops_early_fails_the_stream_once_sync() -> None:
    publisher = RecordingPublisher()
    stream = publish(weather_agent().run("Weather?", stream=True, stream_events=True), publisher)
    for event in stream:
        if event.event == "ToolCallStarted":
            break
    stream.close()
    assert names(publisher)[-1] == "ToolCallStarted"
    assert publisher.outcome == "fail" and publisher.close_calls == 1


async def test_a_caller_that_stops_early_fails_the_stream_once_async() -> None:
    publisher = AsyncRecordingPublisher()
    stream = apublish(weather_agent().arun("Weather?", stream=True, stream_events=True), publisher)
    async for event in stream:
        if event.event == "ToolCallStarted":
            break
    await stream.aclose()  # type: ignore[attr-defined]
    assert names(publisher)[-1] == "ToolCallStarted"
    assert publisher.outcome == "fail" and publisher.close_calls == 1


async def test_a_task_cancelled_while_it_waits_for_the_next_event_fails_the_stream() -> None:
    SLOW["seconds"] = 1.0
    publisher = AsyncRecordingPublisher()
    tool_started = asyncio.Event()

    async def consume() -> None:
        stream = weather_agent(tool_name="slow_weather").arun("Weather?", stream=True, stream_events=True)
        async for event in apublish(stream, publisher):
            if event.event == "ToolCallStarted":
                tool_started.set()

    task = asyncio.create_task(consume())
    await asyncio.wait_for(tool_started.wait(), 5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert names(publisher)[-1] == "ToolCallStarted"
    assert publisher.outcome == "fail" and publisher.close_calls == 1  # awaited: the close completed


@pytest.mark.parametrize(("cancelled", "outcome"), [("fail", "fail"), ("end", "end")])
async def test_cancel_run_during_a_tool_ends_with_run_cancelled(cancelled: Outcome, outcome: Outcome) -> None:
    SLOW["seconds"] = 0.3
    publisher = AsyncRecordingPublisher()
    agent = weather_agent(tool_name="slow_weather")
    run_id = f"run-to-cancel-{cancelled}"
    stream = agent.arun("Weather?", stream=True, stream_events=True, run_id=run_id)
    async for event in apublish(stream, publisher, cancelled=cancelled):
        if event.event == "ToolCallStarted":
            assert Agent.cancel_run(run_id)
    assert "RunCancelled" in names(publisher) and names(publisher)[-1] in ("RunCancelled", "RunCompleted")
    assert publisher.outcome == outcome


async def test_a_paused_run_ends_its_stream_and_continues_on_another_with_the_same_run_id() -> None:
    agent = weather_agent(tool_name="book_trip", answer="Your trip is booked.", db=InMemoryDb())
    first = AsyncRecordingPublisher()
    paused: Any = None
    async for event in apublish(agent.arun("Book Paris", stream=True, stream_events=True, session_id="s-hitl"), first):
        if event.event == "RunPaused":
            paused = event
    assert paused is not None and names(first)[-1] == "RunPaused" and first.outcome == "end"
    assert payloads(first)[-1]["tools"][0]["requires_confirmation"] is True

    for requirement in paused.active_requirements:
        requirement.confirm()
    second = AsyncRecordingPublisher()
    resumed = agent.acontinue_run(
        run_id=paused.run_id,
        session_id="s-hitl",
        requirements=paused.requirements,
        stream=True,
        stream_events=True,
    )
    await collect(apublish(resumed, second))
    assert names(second)[0] == "RunContinued" and names(second)[-1] == "RunCompleted"
    assert {payload["run_id"] for payload in payloads(second)} == {paused.run_id}
    assert "booked Paris" in "".join(second.data) and second.outcome == "end"


def team(member_fails: bool = False) -> Team:
    member = Agent(
        id="weather-member",
        name="Weather Member",
        role="weather",
        model=ScriptedModel(fail_on_turn=2 if member_fails else None),
        tools=[get_weather],
        telemetry=False,
    )
    leader = ScriptedModel(
        tool_name="delegate_task_to_member",
        tool_args={"member_id": "weather-member", "task": "weather in Paris"},
        answer="The member says it is sunny.",
    )
    return Team(id="weather-team", name="Team", model=leader, members=[member], telemetry=False)


async def test_a_team_run_carries_its_members_events_and_ends_with_the_leader() -> None:
    publisher = AsyncRecordingPublisher()
    await collect(apublish(team().arun("Weather?", stream=True, stream_events=True), publisher))
    rows = payloads(publisher)
    root = rows[0]["run_id"]
    members = [row for row in rows if row.get("agent_id") == "weather-member"]
    assert members and all(row["parent_run_id"] == root and row["run_id"] != root for row in members)
    member_done = next(index for index, row in enumerate(rows) if row["event"] == "RunCompleted")
    assert member_done < len(rows) - 1 and rows[-1]["event"] == "TeamRunCompleted"
    assert publisher.outcome == "end"


async def test_a_member_s_failure_does_not_fail_a_team_that_completes() -> None:
    publisher = AsyncRecordingPublisher()
    await collect(apublish(team(member_fails=True).arun("Weather?", stream=True, stream_events=True), publisher))
    assert "RunError" in names(publisher) and names(publisher)[-1] == "TeamRunCompleted"
    assert publisher.outcome == "end"


async def test_a_workflow_whose_step_fails_and_is_skipped_ends() -> None:
    publisher = AsyncRecordingPublisher()
    step = Step(name="lookup", agent=weather_agent(fail_on_turn=2), max_retries=0)
    workflow = Workflow(id="weather-wf", name="Weather WF", steps=[step], telemetry=False)
    await collect(apublish(workflow.arun("Weather?", stream=True, stream_events=True), publisher))
    rows = payloads(publisher)
    errors = [row for row in rows if row["event"] == "RunError"]
    assert errors and all(row["run_id"] != rows[0]["run_id"] for row in errors)
    assert rows[0]["event"] == "WorkflowStarted" and rows[-1]["event"] == "WorkflowCompleted"
    assert publisher.outcome == "end"


async def test_a_workflow_that_raises_fails_the_stream_and_reaches_the_caller() -> None:
    async def steps(workflow: Workflow, execution_input: Any) -> AsyncIterator[str]:
        raise RuntimeError("workflow blew up")
        yield ""

    publisher = AsyncRecordingPublisher()
    workflow = Workflow(id="failing-wf", name="Failing WF", steps=steps, telemetry=False)
    with pytest.raises(RuntimeError, match="workflow blew up"):
        await collect(apublish(workflow.arun("Go", stream=True, stream_events=True), publisher))
    assert names(publisher)[0] == "WorkflowStarted" and publisher.outcome == "fail"
    errors = [row for row in payloads(publisher) if row["event"] == "WorkflowError"]
    assert all(row["error"] == "An error occurred." for row in errors)  # agno 3 sends one; 2.9 raises at once


class _CheckingPublisher(AsyncRecordingPublisher):
    """Records what the agent's database holds at the moment the close line goes out."""

    def __init__(self, agent: Agent, session_id: str) -> None:
        super().__init__()
        self.agent = agent
        self.session_id = session_id
        self.stored_at_close: Any = None

    async def close(self, outcome: Outcome = "end", timeout: float = 10.0) -> PublishResult:
        run_id = payloads(self)[0]["run_id"]
        self.stored_at_close = await self.agent.aget_run_output(run_id, session_id=self.session_id)
        return await super().close(outcome, timeout)


async def test_the_run_is_stored_before_the_close_line() -> None:
    agent = weather_agent(db=InMemoryDb())
    publisher = _CheckingPublisher(agent, "s-store")
    order: list[str] = []

    async def before_close(summary: RunSummary) -> None:
        order.append(f"{summary.status} {summary.outcome} closed={publisher.outcome is not None}")

    stream = agent.arun("Weather?", stream=True, stream_events=True, session_id="s-store")
    await collect(apublish(stream, publisher, before_close=before_close))
    stored = publisher.stored_at_close
    assert stored is not None and str(getattr(stored.status, "value", stored.status)).lower() == "completed"
    assert str(stored.content).endswith("It is 21 degrees and sunny in Paris.")
    assert order == ["completed end closed=False"] and publisher.outcome == "end"


async def test_a_synchronous_publisher_works_under_apublish() -> None:
    publisher = RecordingPublisher()
    await collect(apublish(weather_agent().arun("Weather?", stream=True, stream_events=True), publisher))
    assert names(publisher)[-1] == "RunCompleted" and publisher.outcome == "end"
