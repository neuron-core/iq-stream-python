"""The data of a frame: LangGraph Server's serialization, trimmed metadata, masked errors, and what happens to a part
that does not fit the frame limit."""

from __future__ import annotations

import datetime
import decimal
import enum
import json
import logging
import math
import pathlib
import uuid
from typing import Any

import pytest
from fakes import LANGGRAPH_1_2, USER, FakeToolCallingModel, hitl_agent, tool_call, weather_agent
from iq_stream.testing import RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import ToolException, tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Interrupt, interrupt
from typing_extensions import TypedDict

from iq_stream_langchain import (
    DEFAULT_STREAM_MODE,
    MASKED_MESSAGE,
    SLIM_METADATA,
    Oversize,
    encode_part,
    error_part,
    publish_run,
    relay,
)
from iq_stream_langchain._encode import TRUNCATED_MARKER

FULL_METADATA = {
    "langgraph_node": "model",
    "langgraph_step": 1,
    "langgraph_checkpoint_ns": "model:1",
    "langgraph_path": ["__pregel_pull", "model"],
    "langgraph_triggers": ["branch:to:model"],
    "ls_provider": "fake",
    "ls_model_name": "fake-1",
    "lc_agent_name": "weather",
}


def token_part(**metadata: Any) -> dict[str, Any]:
    return {"type": "messages", "ns": (), "data": (AIMessageChunk(content="Hi", id="m1"), metadata or FULL_METADATA)}


def test_a_messages_part_keeps_the_metadata_the_clients_read() -> None:
    part = json.loads(encode_part(token_part()))
    assert part["type"] == "messages" and part["ns"] == []
    message, metadata = part["data"]
    assert message["content"] == "Hi" and message["type"] == "AIMessageChunk" and message["id"] == "m1"
    assert set(metadata) == {
        "langgraph_node",
        "langgraph_step",
        "langgraph_checkpoint_ns",
        "ls_model_name",
        "lc_agent_name",
    }
    assert set(metadata) <= set(SLIM_METADATA)


def test_metadata_full_or_listed() -> None:
    assert json.loads(encode_part(token_part(), metadata="full"))["data"][1] == FULL_METADATA
    assert json.loads(encode_part(token_part(), metadata=["ls_provider"]))["data"][1] == {"ls_provider": "fake"}


def test_slim_metadata_in_a_run() -> None:
    publisher = RecordingPublisher()
    with publisher:
        list(relay(publisher, weather_agent(), USER))
    slim = [json.loads(data)["data"][1] for data in publisher.data if json.loads(data)["type"] == "messages"]
    assert all(set(metadata) <= set(SLIM_METADATA) for metadata in slim)
    assert all("langgraph_step" in metadata for metadata in slim)

    publisher = RecordingPublisher()
    with publisher:
        list(relay(publisher, weather_agent(), USER, metadata="full"))
    full = [json.loads(data)["data"][1] for data in publisher.data if json.loads(data)["type"] == "messages"]
    assert any("langgraph_triggers" in metadata for metadata in full)


def test_interrupts_and_dataclasses_are_encoded() -> None:
    part = {
        "type": "updates",
        "ns": ("tools:abc",),
        "data": {"__interrupt__": (Interrupt(value={"question": "ok?"}, id="i1"),)},
    }
    schema = {"response_schema": None} if LANGGRAPH_1_2 else {}  # a field of langgraph 1.2's Interrupt
    assert json.loads(encode_part(part)) == {
        "type": "updates",
        "ns": ["tools:abc"],
        "data": {"__interrupt__": [{"value": {"question": "ok?"}, "id": "i1", **schema}]},
    }
    values = {"type": "values", "ns": (), "data": {"x": 1}, "interrupts": (Interrupt(value=1, id="i2"),)}
    assert json.loads(encode_part(values))["interrupts"] == [{"value": 1, "id": "i2", **schema}]
    assert "interrupts" not in json.loads(encode_part({**values, "interrupts": ()}))


def test_an_error_part_names_the_error_and_masks_its_message() -> None:
    assert json.loads(error_part(ValueError("secret detail"))) == {
        "type": "error",
        "ns": [],
        "data": {"error": "ValueError", "message": MASKED_MESSAGE},
    }
    assert json.loads(error_part(ValueError("secret detail"), expose=True))["data"]["message"] == "secret detail"


def test_a_failing_tool_has_its_message_masked() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        raise ValueError("the API key sk-123 was refused")

    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")])])  # type: ignore[list-item]
    )
    publisher = RecordingPublisher()
    with pytest.raises(ValueError, match="sk-123"), publisher:
        publish_run(publisher, create_agent(model, [get_weather]), USER)
    assert not any("sk-123" in data for data in publisher.data)
    if LANGGRAPH_1_2:  # the tools stream mode
        tool_error = next(json.loads(data)["data"] for data in publisher.data if '"tool-error"' in data)
        assert tool_error == {"event": "tool-error", "tool_call_id": "c1", "message": MASKED_MESSAGE}
    assert json.loads(publisher.data[-1])["data"] == {"error": "ValueError", "message": MASKED_MESSAGE}
    assert publisher.outcome == "fail"

    part = {"type": "tools", "ns": (), "data": {"event": "tool-error", "tool_call_id": "c1", "message": "sk-123"}}
    assert json.loads(encode_part(part, expose_errors=True))["data"]["message"] == "sk-123"


def big_tool_agent(size: int) -> Any:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        return "é" + "x" * size + '"'

    model = FakeToolCallingModel(
        messages=iter(
            [AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Done."]  # type: ignore[list-item]
        )
    )
    return create_agent(model, [get_weather])


def test_a_large_tool_result_is_truncated_and_a_large_update_not_sent() -> None:
    publisher = RecordingPublisher(max_frame_bytes=2048)
    reports: list[Oversize] = []
    with publisher:
        final = publish_run(publisher, big_tool_agent(100_000), USER, on_oversize=reports.append)

    assert final["messages"][-1].content == "Done.", "the run completed"
    assert len(final["messages"][-2].content) == 100_002, "the state holds the whole result"
    parts = [json.loads(data) for data in publisher.data]
    assert all(publisher.frame_size(data) <= 2048 for data in publisher.data)
    tool_message = next(part for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "tool")
    content = tool_message["data"][0]["content"]
    assert content.startswith("éxxx") and content.endswith("…[truncated]")
    assert tool_message["data"][1]["truncated"] is True
    assert publisher.frame_size(publisher.data[parts.index(tool_message)]) > 2000, "cut as little as possible"
    # The tools node's update and the tool's finish event carry the result too: not sent.
    assert [part["data"]["event"] for part in parts if part["type"] == "tools"] == (["tool-started"] * LANGGRAPH_1_2)
    assert [list(part["data"]) for part in parts if part["type"] == "updates"] == [["model"], ["model"]]
    assert [(report.part_type, report.truncated) for report in reports] == [
        *([("tools", False)] * LANGGRAPH_1_2),
        ("messages", True),
        ("updates", False),
    ]
    assert all(report.size > 100_000 and report.limit == 2048 for report in reports)
    assert publisher.outcome == "end"


def test_an_oversized_part_is_reported_once_per_run(caplog: pytest.LogCaptureFixture) -> None:
    publisher = RecordingPublisher(max_frame_bytes=2048)
    with caplog.at_level(logging.WARNING, logger="iq_stream_langchain"), publisher:
        publish_run(publisher, big_tool_agent(10_000), USER)
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1
    kind, what = ("tools", "not sent") if LANGGRAPH_1_2 else ("messages", "truncated")
    assert f"{kind} part of" in messages[0] and "2048" in messages[0] and what in messages[0]
    assert "xxx" not in messages[0], "no frame data in a log line"


def test_a_message_without_text_content_that_does_not_fit_is_not_sent() -> None:
    publisher = RecordingPublisher(max_frame_bytes=1024)
    reports: list[Oversize] = []
    blocks = [{"type": "text", "text": "y" * 5000}]
    model = FakeToolCallingModel(messages=iter([AIMessage(content=blocks)]))  # type: ignore[list-item]
    with publisher:
        final = publish_run(
            publisher, create_agent(model, []), USER, stream_mode=("updates",), on_oversize=reports.append
        )
    assert final["messages"][-1].content == blocks
    assert [json.loads(data)["type"] for data in publisher.data] == ["metadata"]
    assert reports == [Oversize("updates", reports[0].size, 1024, truncated=False)]


def interrupting_agent(value: Any) -> Any:
    @tool
    def book(city: str) -> str:
        """Book a trip; a human approves it first."""
        return f"booked: {interrupt(value)}"

    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("book", {"city": "Turin"}, "c1")])])  # type: ignore[list-item]
    )
    return create_agent(model, [book], checkpointer=InMemorySaver())


def test_an_interrupt_over_the_limit_is_cut_and_never_dropped() -> None:
    """It is the only part that tells the page the run waits for a human."""
    config: Any = {"configurable": {"thread_id": "big-draft"}}
    publisher = RecordingPublisher(max_frame_bytes=1024)
    reports: list[Oversize] = []
    with publisher:
        final = publish_run(
            publisher,
            interrupting_agent({"question": "send it?", "draft": "d" * 1800}),
            USER,
            config,
            on_oversize=reports.append,
        )
    last = json.loads(publisher.data[-1])
    assert last["type"] == "updates"
    [cut] = last["data"]["__interrupt__"]
    assert cut["id"] == final["__interrupt__"][0].id, "the id resumes the run"
    assert cut["truncated"] is True
    assert cut["value"]["question"] == "send it?", "the structure and the short strings stay"
    assert cut["value"]["draft"].startswith("ddd") and cut["value"]["draft"].endswith(TRUNCATED_MARKER)
    assert 900 < publisher.frame_size(publisher.data[-1]) <= 1024, "cut as little as possible"
    assert reports == [Oversize("updates", reports[0].size, 1024, truncated=True)]
    assert publisher.outcome == "end"

    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publisher:
        publish_run(publisher, interrupting_agent("Approve this draft: " + "d" * 3000), USER, config)
    value = json.loads(publisher.data[-1])["data"]["__interrupt__"][0]["value"]
    assert value.startswith("Approve this draft: ddd") and value.endswith(TRUNCATED_MARKER)
    assert 900 < publisher.frame_size(publisher.data[-1]) <= 1024, "cut as little as possible"


def test_a_human_in_the_loop_interrupt_keeps_what_a_client_needs_to_ask() -> None:
    """HumanInTheLoopMiddleware's value repeats the tool's arguments; @ai-sdk/langchain reads action_requests."""
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publisher:
        final = publish_run(publisher, hitl_agent(), USER, {"configurable": {"thread_id": "hitl"}})
    [cut] = json.loads(publisher.data[-1])["data"]["__interrupt__"]
    assert cut["id"] == final["__interrupt__"][0].id
    [request] = cut["value"]["action_requests"]
    assert request["name"] == "send_email"
    assert request["args"]["to"] == "team@example.com"
    assert request["args"]["body"].endswith(TRUNCATED_MARKER) and request["args"]["body"].startswith("Dear team")
    [review] = cut["value"]["review_configs"]
    assert review["action_name"] == "send_email" and "approve" in review["allowed_decisions"]
    assert publisher.frame_size(publisher.data[-1]) <= 1024


def test_an_interrupt_whose_structure_does_not_fit_keeps_its_id() -> None:
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with publisher:
        final = publish_run(
            publisher,
            interrupting_agent({f"field_{index}": index for index in range(200)}),
            USER,
            {"configurable": {"thread_id": "wide"}},
        )
    assert json.loads(publisher.data[-1])["data"] == {
        "__interrupt__": [{"value": None, "id": final["__interrupt__"][0].id, "truncated": True}]
    }


def test_a_failed_tool_result_is_masked_wherever_it_goes() -> None:
    """A tool that reports its own failure (handle_tool_error, an invalid call) returns a ToolMessage whose content is
    the error's text, with status "error"."""

    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        raise ToolException("the API key sk-123 was refused")

    get_weather.handle_tool_error = True

    def agent() -> Any:
        model = FakeToolCallingModel(
            messages=iter(
                [AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sorry."]  # type: ignore[list-item]
            )
        )
        return create_agent(model, [get_weather])

    publisher = RecordingPublisher()
    with publisher:
        final = publish_run(publisher, agent(), USER, stream_mode=(*DEFAULT_STREAM_MODE, "values", "tasks"))
    assert final["messages"][-2].content == "the API key sk-123 was refused", "the state keeps it"
    assert not any("sk-123" in data for data in publisher.data)
    parts = [json.loads(data) for data in publisher.data]
    tool_message = next(part for part in parts if part["type"] == "messages" and part["data"][0]["type"] == "tool")
    assert tool_message["data"][0]["content"] == MASKED_MESSAGE and tool_message["data"][0]["status"] == "error"
    update = next(part for part in parts if part["type"] == "updates" and "tools" in part["data"])
    assert update["data"]["tools"]["messages"][0]["content"] == MASKED_MESSAGE
    assert publisher.outcome == "end"

    exposed = RecordingPublisher()
    with exposed:
        publish_run(exposed, agent(), USER, expose_errors=True)
    assert any("sk-123" in data for data in exposed.data)


def test_non_finite_floats_are_null() -> None:
    """A browser's JSON.parse refuses NaN and Infinity, and the whole stream with them."""
    part = {"type": "custom", "ns": (), "data": {"score": math.nan, "max": math.inf, "min": -math.inf, "ok": 1.5}}

    def refuse(constant: str) -> None:
        raise ValueError(constant)

    frame = encode_part(part)
    assert json.loads(frame, parse_constant=refuse) == {
        "type": "custom",
        "ns": [],
        "data": {"score": None, "max": None, "min": None, "ok": 1.5},
    }
    message = {
        "type": "messages",
        "ns": (),
        "data": (AIMessageChunk(content="Hi", response_metadata={"p": math.nan}), {}),
    }
    assert json.loads(encode_part(message), parse_constant=refuse)["data"][0]["response_metadata"] == {"p": None}


class Status(enum.Enum):
    DONE = "done"


def test_standard_types_are_written_as_langgraph_server_writes_them() -> None:
    at = datetime.datetime(2026, 11, 14, 21, 0, tzinfo=datetime.timezone.utc)
    part = {
        "type": "custom",
        "ns": (),
        "data": {
            "at": at,
            "day": at.date(),
            "time": datetime.time(21, 30),
            "took": datetime.timedelta(seconds=1.5),
            "id": uuid.UUID(int=1),
            "status": Status.DONE,
            "price": decimal.Decimal("10.50"),
            "path": pathlib.PurePosixPath("/tmp/x"),
        },
    }
    assert json.loads(encode_part(part))["data"] == {
        "at": "2026-11-14T21:00:00+00:00",
        "day": "2026-11-14",
        "time": "21:30:00",
        "took": 1.5,
        "id": "00000000-0000-0000-0000-000000000001",
        "status": "done",
        "price": "10.50",
        "path": "/tmp/x",
    }
    message = {"type": "messages", "ns": (), "data": (AIMessage(content="Hi", response_metadata={"created": at}), {})}
    assert json.loads(encode_part(message))["data"][0]["response_metadata"] == {"created": "2026-11-14T21:00:00+00:00"}


def test_an_on_oversize_hook_that_raises_never_breaks_the_run(caplog: pytest.LogCaptureFixture) -> None:
    def broken(report: Oversize) -> None:
        raise ConnectionError("statsd is down at 10.0.0.1")

    publisher = RecordingPublisher(max_frame_bytes=2048)
    with caplog.at_level(logging.WARNING, logger="iq_stream_langchain"), publisher:
        final = publish_run(publisher, big_tool_agent(10_000), USER, on_oversize=broken)
    assert final["messages"][-1].content == "Done."
    assert all(json.loads(data)["type"] != "error" for data in publisher.data)
    assert publisher.outcome == "end"
    messages = [record.getMessage() for record in caplog.records]
    assert len(messages) == 1 and "on_oversize raised ConnectionError" in messages[0]
    assert "statsd" not in messages[0]


class State(TypedDict):
    value: Any


def test_an_exposed_error_too_large_for_a_frame_is_sent_masked() -> None:
    def node(state: State) -> dict[str, Any]:
        raise RuntimeError("x" * 2000)

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with pytest.raises(RuntimeError), publisher:
        publish_run(publisher, graph.compile(), {"value": None}, expose_errors=True)
    assert json.loads(publisher.data[-1])["data"] == {"error": "RuntimeError", "message": MASKED_MESSAGE}
    assert publisher.outcome == "fail"


def test_the_error_of_a_task_result_is_masked() -> None:
    def node(state: State) -> dict[str, Any]:
        raise ValueError("the API key sk-123 was refused")

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    for expose, error in ((False, MASKED_MESSAGE), (True, {"error": "ValueError"})):
        publisher = RecordingPublisher()
        with pytest.raises(ValueError), publisher:
            publish_run(publisher, graph.compile(), {"value": None}, stream_mode=("tasks",), expose_errors=expose)
        results = [json.loads(data)["data"] for data in publisher.data if json.loads(data)["type"] == "tasks"]
        assert [result["error"] for result in results if result.get("error") is not None] == [error]
