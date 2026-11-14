"""Publishing never breaks the run: a publisher that is inert, or that gives up halfway, costs frames and nothing
else, and the run is read to its end."""

from __future__ import annotations

import json
import logging
from typing import Any

import pytest
from fakes import FINAL, USER, weather_agent
from iq_stream import IQError
from iq_stream.testing import AsyncRecordingPublisher, RecordingPublisher
from langchain_core.messages import AIMessage
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

import iq_stream_langchain._relay as relay_module
from iq_stream_langchain import apublish_run, publish_run, relay


def test_an_inert_publisher_encodes_nothing_and_the_run_completes(monkeypatch: pytest.MonkeyPatch) -> None:
    def no_encoding(*args: object, **kwargs: object) -> str:
        raise AssertionError("a part was encoded for a publisher that gave up")

    monkeypatch.setattr(relay_module, "shape", no_encoding)
    monkeypatch.setattr(relay_module, "dumps", no_encoding)
    publisher = RecordingPublisher()
    publisher.fail()
    with publisher:
        final = publish_run(publisher, weather_agent(), USER)
    assert final["messages"][-1].content == FINAL
    assert publisher.frames == []


async def test_an_inert_async_publisher_leaves_the_run_alone() -> None:
    publisher = AsyncRecordingPublisher()
    publisher.fail()
    async with publisher:
        final = await apublish_run(publisher, weather_agent(), USER)
    assert final["messages"][-1].content == FINAL
    assert publisher.frames == []


def test_a_publisher_that_gives_up_halfway_stops_frames_and_the_run_goes_on() -> None:
    publisher = RecordingPublisher()
    seen = 0
    with publisher:
        for _ in relay(publisher, weather_agent(), USER):
            seen += 1
            if seen == 5:
                publisher.fail()  # as after a fatal answer from the server
    assert seen == 22, "the whole run was read"
    assert len(publisher.frames) == 6, "the metadata part and five parts"
    assert publisher.outcome is None, "a publisher that gave up has no close line of its own"


def test_a_failed_run_on_an_inert_publisher_still_raises() -> None:
    from fakes import failing_agent

    publisher = RecordingPublisher()
    publisher.fail()
    with pytest.raises(RuntimeError, match="exploded"), publisher:
        publish_run(publisher, failing_agent(), USER)
    assert publisher.frames == []


def test_with_on_error_raise_the_publishing_error_leaves_the_relay() -> None:
    publisher = RecordingPublisher(on_error="raise")
    seen = 0
    with pytest.raises(IQError, match="give up"), publisher:  # noqa: PT012
        for _ in relay(publisher, weather_agent(), USER):
            seen += 1
            if seen == 3:
                publisher.fail()
    assert seen == 3, "the run stops with the error"
    assert len(publisher.frames) == 4
    assert all(json.loads(data)["type"] != "error" for data in publisher.data)


class State(TypedDict):
    value: Any


class Unencodable:
    def model_dump(self) -> dict[str, Any]:
        raise RuntimeError("cannot be dumped")


def test_a_part_that_cannot_be_encoded_is_dropped_once_reported(caplog: pytest.LogCaptureFixture) -> None:
    def node(state: State) -> dict[str, Any]:
        writer = get_stream_writer()
        writer(Unencodable())
        writer(Unencodable())
        writer({"after": "the bad ones"})
        return {"value": "done"}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    publisher = RecordingPublisher()
    with caplog.at_level(logging.WARNING, logger="iq_stream_langchain"), publisher:
        final = publish_run(publisher, graph.compile(), {"value": None})
    assert final == {"value": "done"}
    custom = [json.loads(data)["data"] for data in publisher.data if json.loads(data)["type"] == "custom"]
    assert custom == [{"after": "the bad ones"}]
    warnings = [record.getMessage() for record in caplog.records]
    assert len(warnings) == 1 and "custom part could not be encoded" in warnings[0]
    assert "cannot be dumped" not in warnings[0]
    assert publisher.outcome == "end"


def test_an_odd_value_is_encoded_by_repr_and_not_lost() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque 42>"

    def node(state: State) -> dict[str, Any]:
        get_stream_writer()({"thing": Opaque(), "tags": {"a"}, "message": AIMessage(content="hé")})
        return {"value": 1}

    graph = StateGraph(State)
    graph.add_node("node", node)
    graph.add_edge(START, "node")
    graph.add_edge("node", END)
    publisher = RecordingPublisher()
    with publisher:
        publish_run(publisher, graph.compile(), {"value": None})
    frame = next(data for data in publisher.data if json.loads(data)["type"] == "custom")
    data = json.loads(frame)["data"]
    assert data["thing"] == "<Opaque 42>"
    assert data["tags"] == ["a"]
    assert data["message"]["content"] == "hé" and data["message"]["type"] == "ai"
    assert "hé" in frame, "non-ASCII text stays unescaped"
