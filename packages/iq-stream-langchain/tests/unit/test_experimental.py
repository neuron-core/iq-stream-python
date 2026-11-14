"""The StreamTransformer for stream_events(version="v3"): it publishes while the application reads only run.output."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest
from fakes import FINAL, USER, failing_agent, weather_agent
from iq_stream.testing import RecordingPublisher

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
