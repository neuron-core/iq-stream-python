"""The data of a frame: LangGraph Server's serialization, trimmed metadata, masked errors, and what happens to a part
that does not fit the frame limit."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fakes import USER, FakeToolCallingModel, tool_call, weather_agent
from iq_stream.testing import RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.tools import tool
from langgraph.types import Interrupt

from iq_stream_langchain import MASKED_MESSAGE, SLIM_METADATA, Oversize, encode_part, error_part, publish_run, relay

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
    assert json.loads(encode_part(part)) == {
        "type": "updates",
        "ns": ["tools:abc"],
        "data": {"__interrupt__": [{"value": {"question": "ok?"}, "id": "i1", "response_schema": None}]},
    }
    values = {"type": "values", "ns": (), "data": {"x": 1}, "interrupts": (Interrupt(value=1, id="i2"),)}
    assert json.loads(encode_part(values))["interrupts"] == [{"value": 1, "id": "i2", "response_schema": None}]
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
    assert [part["data"]["event"] for part in parts if part["type"] == "tools"] == ["tool-started"]
    assert [list(part["data"]) for part in parts if part["type"] == "updates"] == [["model"], ["model"]]
    assert [(report.part_type, report.truncated) for report in reports] == [
        ("tools", False),
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
    assert "tools part of" in messages[0] and "2048" in messages[0] and "not sent" in messages[0]
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
