"""The StreamTransformer for stream_events(version="v3"): it publishes while the application reads only run.output."""

from __future__ import annotations

import json
import logging
from collections import Counter
from typing import Any

import pytest
from fakes import FINAL, USER, FakeToolCallingModel, failing_agent, tool_call, weather_agent
from iq_stream.testing import RecordingPublisher
from langchain.agents import create_agent
from langchain_core.messages import AIMessage
from langchain_core.tools import ToolException, tool
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from iq_stream_langchain import MASKED_MESSAGE, Oversize
from iq_stream_langchain.experimental import iq_transformer

pytestmark = pytest.mark.filterwarnings(
    "ignore:The v3 streaming protocol on Pregel is experimental:langchain_core._api.beta_decorator.LangChainBetaWarning"
)


def kinds(publisher: RecordingPublisher) -> Counter[tuple[str, Any]]:
    events = [json.loads(data) for data in publisher.data]
    assert all(event["type"] == "event" for event in events)
    return Counter(
        (event["method"], event["params"]["data"].get("event") if isinstance(event["params"]["data"], dict) else None)
        for event in events
    )


def test_a_run_read_only_through_its_output_is_published() -> None:
    publisher = RecordingPublisher()
    with publisher:
        run = weather_agent().stream_events(USER, version="v3", transformers=[iq_transformer(publisher)])
        final = run.output
        assert publisher.outcome is None, "the transformer does not close"
    assert final["messages"][-1].text == FINAL
    seen = kinds(publisher)
    for needed in ("message-start", "content-block-delta", "message-finish"):
        assert seen["messages", needed] >= 1
    assert seen["tools", "tool-started"] == seen["tools", "tool-finished"] == 1
    assert seen["updates", None] == 3
    first = json.loads(publisher.data[0])
    assert set(first["params"]) == {"namespace", "timestamp", "data"}
    assert first["params"]["data"]["event"] == "message-start", "the metadata half of a messages payload is left out"
    assert publisher.outcome == "end"


def test_methods_choose_what_is_published() -> None:
    publisher = RecordingPublisher()
    with publisher:
        run = weather_agent().stream_events(
            USER, version="v3", transformers=[iq_transformer(publisher, methods=["tools"])]
        )
        assert run.output
    assert set(kinds(publisher)) == {("tools", "tool-started"), ("tools", "tool-finished")}


def test_a_failing_run_publishes_its_masked_error_and_the_block_closes_fail() -> None:
    publisher = RecordingPublisher()
    with pytest.raises(RuntimeError, match="exploded"), publisher:
        assert failing_agent().stream_events(USER, version="v3", transformers=[iq_transformer(publisher)]).output
    assert json.loads(publisher.data[-1]) == {
        "type": "error",
        "ns": [],
        "data": {"error": "RuntimeError", "message": MASKED_MESSAGE},
    }
    assert publisher.outcome == "fail"


def test_an_event_over_the_limit_is_not_sent() -> None:
    publisher = RecordingPublisher(max_frame_bytes=200)
    reports: list[Oversize] = []
    with publisher:
        transformer = iq_transformer(publisher, methods=["updates"], on_oversize=reports.append)
        assert weather_agent().stream_events(USER, version="v3", transformers=[transformer]).output
    assert publisher.frames == []
    assert [report.part_type for report in reports] == ["updates"] * 3


class State(TypedDict):
    value: Any


def writer_graph(*values: Any) -> Any:
    def node(state: State) -> dict[str, Any]:
        writer = get_stream_writer()
        for value in values:
            writer(value)
        return {"value": "done"}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    return graph.compile()


def test_an_event_that_cannot_be_encoded_is_dropped_and_the_run_completes(caplog: pytest.LogCaptureFixture) -> None:
    circular: dict[str, Any] = {"name": "root"}
    circular["parent"] = circular
    graph = writer_graph({("Turin", "2026-11-14"): 21.0}, circular, {"after": "the bad ones"})
    publisher = RecordingPublisher()
    with caplog.at_level(logging.WARNING, logger="iq_stream_langchain"), publisher:
        run = graph.stream_events({"value": None}, version="v3", transformers=[iq_transformer(publisher)])
        assert run.output == {"value": "done"}
    custom = [json.loads(data)["params"]["data"] for data in publisher.data if json.loads(data)["method"] == "custom"]
    assert custom == [{"after": "the bad ones"}]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1 and "custom event could not be encoded" in warnings[0]
    assert publisher.outcome == "end"


def test_a_failing_tool_has_its_message_masked() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        raise ValueError("the API key sk-123 was refused")

    def agent() -> Any:
        model = FakeToolCallingModel(
            messages=iter([AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")])])  # type: ignore[list-item]
        )
        return create_agent(model, [get_weather])

    publisher = RecordingPublisher()
    with pytest.raises(ValueError), publisher:
        assert agent().stream_events(USER, version="v3", transformers=[iq_transformer(publisher)]).output
    assert not any("sk-123" in data for data in publisher.data)
    tool_error = next(json.loads(data)["params"]["data"] for data in publisher.data if '"tool-error"' in data)
    assert tool_error["message"] == MASKED_MESSAGE

    exposed = RecordingPublisher()
    with pytest.raises(ValueError), exposed:
        transformer = iq_transformer(exposed, expose_errors=True)
        assert agent().stream_events(USER, version="v3", transformers=[transformer]).output
    assert any('"tool-error"' in data and "sk-123" in data for data in exposed.data)


def test_an_on_oversize_hook_that_raises_never_breaks_the_run() -> None:
    def broken(report: Oversize) -> None:
        raise ConnectionError("statsd is down")

    publisher = RecordingPublisher(max_frame_bytes=200)
    with publisher:
        transformer = iq_transformer(publisher, methods=["updates"], on_oversize=broken)
        final = weather_agent().stream_events(USER, version="v3", transformers=[transformer]).output
    assert final["messages"][-1].text == FINAL
    assert publisher.outcome == "end"


def test_an_exposed_error_too_large_for_a_frame_is_sent_masked() -> None:
    def node(state: State) -> dict[str, Any]:
        raise RuntimeError("the provider refused the request: " + "x" * 2000)

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    publisher = RecordingPublisher(max_frame_bytes=1024)
    with pytest.raises(RuntimeError), publisher:
        transformer = iq_transformer(publisher, methods=["custom"], expose_errors=True)
        assert graph.compile().stream_events({"value": None}, version="v3", transformers=[transformer]).output
    assert [json.loads(data) for data in publisher.data] == [
        {"type": "error", "ns": [], "data": {"error": "RuntimeError", "message": MASKED_MESSAGE}}
    ]
    assert publisher.outcome == "fail"


def test_a_failed_tool_result_is_masked() -> None:
    @tool
    def get_weather(city: str) -> str:
        """Get the weather for a city."""
        raise ToolException("the API key sk-123 was refused")

    get_weather.handle_tool_error = True
    model = FakeToolCallingModel(
        messages=iter([AIMessage(content="", tool_calls=[tool_call("get_weather", {"city": "Turin"}, "c1")]), "Sorry."])  # type: ignore[list-item]
    )
    publisher = RecordingPublisher()
    with publisher:
        run = create_agent(model, [get_weather]).stream_events(
            USER, version="v3", transformers=[iq_transformer(publisher, methods=["updates", "tools"])]
        )
        assert run.output["messages"][-2].content == "the API key sk-123 was refused"
    assert publisher.data and not any("sk-123" in data for data in publisher.data)
    assert any(MASKED_MESSAGE in data for data in publisher.data)
