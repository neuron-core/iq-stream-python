"""protocol="crewai": CrewAI's own events, one named frame each."""

from __future__ import annotations

import json
from typing import Any

import pytest
from crew_fakes import ANSWER, TERM, FakeStreamingLLM, LookupTool, make_crew
from crewai import Agent, Crew, Task
from iq_stream.testing import RecordingPublisher

from iq_stream_crewai import EVENT_TYPES, publish


def frames(publisher: RecordingPublisher) -> list[tuple[str | None, dict[str, Any]]]:
    return [(frame.event, json.loads(frame.data)) for frame in publisher.frames]


def test_event_names_and_members() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai") as run:
        make_crew().kickoff()
    seen = frames(publisher)
    names = [name for name, _ in seen]
    assert names == [
        "crew_kickoff_started",
        "task_started",
        "agent_execution_started",
        "llm_call_started",
        *["llm_stream_chunk"] * 6,
        "llm_call_completed",
        "tool_usage_started",
        "tool_usage_finished",
        "llm_call_started",
        *["llm_stream_chunk"] * 6,
        "llm_call_completed",
        "agent_execution_completed",
        "task_completed",
        "crew_kickoff_completed",
    ]
    assert set(names) <= set(EVENT_TYPES)
    by_name = {name: data for name, data in seen}
    assert by_name["crew_kickoff_started"] == {"seq": 1, "crew_name": "crew-demo"}
    assert set(by_name["agent_execution_started"]) == {"seq", "agent_id", "agent_role", "task_id"}
    assert set(by_name["llm_call_started"]) == {"seq", "call_id", "model", "agent_id", "agent_role", "task_id"}
    assert by_name["tool_usage_started"]["tool_args"] == {"term": TERM}
    assert by_name["tool_usage_finished"]["output"] == f"value:{TERM}"
    assert by_name["tool_usage_finished"]["from_cache"] is False
    assert by_name["task_completed"]["output"] == ANSWER
    assert by_name["crew_kickoff_completed"]["output"] == ANSWER
    chunks = [data for name, data in seen if name == "llm_stream_chunk"]
    # A tool call delta carries the call's id, name and index, never the accumulated arguments.
    tool_call = {"id": chunks[0]["tool_call"]["id"], "name": "lookup", "index": 0}
    assert chunks[0] == {"seq": chunks[0]["seq"], "call_id": chunks[0]["call_id"], "chunk": "", "tool_call": tool_call}
    assert json.loads("".join(chunk["chunk"] for chunk in chunks[:6])) == {"term": TERM}
    assert "".join(chunk["chunk"] for chunk in chunks[6:]) == ANSWER
    assert all(set(chunk) <= {"seq", "call_id", "chunk", "tool_call"} for chunk in chunks)
    assert all("arguments" not in frame.data and "null" not in frame.data for frame in publisher.frames)
    seqs = [data["seq"] for _, data in seen]
    assert seqs == sorted(seqs)
    assert publisher.outcome == "end"
    assert run.errors == []


def test_errors_are_masked_unless_exposed() -> None:
    for expose in (False, True):
        publisher = RecordingPublisher()
        with pytest.raises(RuntimeError), publish(publisher, protocol="crewai", expose_errors=expose):
            make_crew(fail_on_call=2, tool=LookupTool(explode=True)).kickoff()
        errors = {frame.event: json.loads(frame.data)["error"] for frame in publisher.frames if "error" in frame.data}
        failed = {"tool_usage_error", "llm_call_failed", "agent_execution_error", "task_failed", "crew_kickoff_failed"}
        assert set(errors) == failed
        if expose:
            assert "fake provider outage at sk-secret" in errors["llm_call_failed"]
            assert "lookup backend is down" in errors["tool_usage_error"]
        else:
            assert set(errors.values()) == {"An error occurred."}
            assert "secret" not in "".join(publisher.data)
        # The native protocol has no final event of its own: the close line says how the run ended.
        assert publisher.frames[-1].event == "crew_kickoff_failed"
        assert publisher.outcome == "fail"


def test_long_text_values_are_cut_to_max_field_bytes() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai", max_field_bytes=10):
        make_crew(tool=LookupTool(result="r" * 50)).kickoff()
    [finished] = [json.loads(frame.data) for frame in publisher.frames if frame.event == "tool_usage_finished"]
    assert finished["output"] == "r" * 10 + "… [40 bytes truncated]"
    # The answer streamed in chunks is not cut.
    chunks = [json.loads(frame.data)["chunk"] for frame in publisher.frames if frame.event == "llm_stream_chunk"]
    assert "".join(chunks).endswith(ANSWER)


def test_max_field_bytes_counts_bytes_of_utf_8_and_never_cuts_a_character_in_two() -> None:
    publisher = RecordingPublisher()
    with publish(publisher, protocol="crewai", max_field_bytes=7):
        make_crew(tool=LookupTool(result="é" * 10 + "😀")).kickoff()  # 2 bytes each, then 4
    [finished] = [json.loads(frame.data) for frame in publisher.frames if frame.event == "tool_usage_finished"]
    assert finished["output"] == "ééé… [18 bytes truncated]"


def test_long_names_are_cut_to_fit_the_frame() -> None:
    llm = FakeStreamingLLM(tool_args=[{"term": TERM}])
    agent = Agent(role="R" * 3000, goal="g", backstory="b", llm=llm, tools=[LookupTool()], max_retry_limit=0)
    crew = Crew(name="C" * 3000, agents=[agent], tasks=[Task(description="d", expected_output="o", agent=agent)])
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publish(publisher, protocol="crewai") as run:
        crew.kickoff()
    for frame in publisher.frames:
        assert publisher.frame_size(frame.data, frame.event) <= 1024
    seen = frames(publisher)
    assert [name for name, _ in seen].count("agent_execution_started") == 1
    roles = [data["agent_role"] for _, data in seen if "agent_role" in data]
    assert len(roles) == 5  # agent started and completed, two model calls, the tool
    assert all(role.startswith("RRR") and role.endswith("bytes truncated]") for role in roles)
    assert seen[0][1]["crew_name"].endswith("bytes truncated]")
    assert run.errors == []


def test_values_are_copied_when_the_event_is_emitted() -> None:
    from crewai.events.types.tool_usage_events import ToolUsageStartedEvent

    from iq_stream_crewai._snapshot import Snapshotter

    event = ToolUsageStartedEvent(tool_name="lookup", tool_args={"term": "x", "seen": ["a"]})
    snapshot = Snapshotter(frozenset(EVENT_TYPES), None, False)(None, event)
    assert snapshot is not None
    assert isinstance(event.tool_args, dict)
    event.tool_args["seen"].append("b")  # the tool, or the agent, changes its arguments after the event
    event.tool_args["term"] = "y"
    assert snapshot.fields["tool_args"] == {"term": "x", "seen": ["a"]}
