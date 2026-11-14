from __future__ import annotations

import json
from collections.abc import Hashable

from haystack.dataclasses import ReasoningContent, StreamingChunk
from haystack.dataclasses.streaming_chunk import ToolCallDelta

from haystack_integrations.iq_stream import NativeConverter
from haystack_integrations.iq_stream.converters.base import frame_size
from tests.fakes import result_chunk, tool_turn, weather_call

MAIN: Hashable = ("main", None)


def convert(conv: NativeConverter, chunk: StreamingChunk) -> list[dict[str, object]]:
    return [json.loads(frame.data) for frame in conv.convert(chunk, MAIN)]


def test_each_chunk_is_one_frame_without_meta_or_provider_extras() -> None:
    call = weather_call()
    conv = NativeConverter()
    frames = [json.loads(f.data) for f in conv.start()]
    for chunk in [*tool_turn([call], shape="anthropic", reasoning=["Hm."]).chunks, result_chunk(call)]:
        frames += convert(conv, chunk)
    frames += [json.loads(f.data) for f in conv.finish()]
    assert frames[0] == {"type": "start"} and frames[-1] == {"type": "finish"}
    text = json.dumps(frames)
    for secret in ("secret-signature", "secret-extra", "meta", "extra", "raw"):
        assert secret not in text
    result = frames[-2]["tool_call_result"]
    assert result == {
        "result": "Sunny, 21C",
        "origin": {"tool_name": "get_weather", "arguments": {"city": "Paris"}, "id": "call_1"},
        "error": False,
    }


def test_errors_are_masked_and_cancellation_has_its_own_frame() -> None:
    assert json.loads(NativeConverter().error(KeyError("secret"))[0].data) == {
        "type": "error",
        "message": "An error occurred.",
    }
    exposed = NativeConverter(expose_errors=True).error(KeyError("k"))
    assert json.loads(exposed[0].data) == {"type": "error", "message": "KeyError: 'k'"}
    assert json.loads(NativeConverter().cancelled()[0].data) == {"type": "cancelled"}
    assert json.loads(NativeConverter().custom("ask", [1])[0].data) == {"type": "custom", "name": "ask", "value": [1]}


def test_oversize_content_reasoning_and_arguments_are_split_into_chunks_of_the_same_kind() -> None:
    conv = NativeConverter(max_frame_bytes=1024)
    text = 'ä"\n' * 2_000
    pieces = convert(conv, StreamingChunk(content=text, index=0, start=True, finish_reason="stop"))
    assert len(pieces) > 1 and "".join(str(p["content"]) for p in pieces) == text
    assert [p["start"] for p in pieces] == [True] + [False] * (len(pieces) - 1)
    assert [p["finish_reason"] for p in pieces] == [None] * (len(pieces) - 1) + ["stop"]

    reasoning = convert(conv, StreamingChunk(content="", index=0, reasoning=ReasoningContent("r" * 5_000)))
    assert "".join(p["reasoning"]["reasoning_text"] for p in reasoning) == "r" * 5_000  # type: ignore[index]

    arguments = json.dumps({"q": "€" * 2_000})
    delta = ToolCallDelta(index=3, tool_name="search", id="c9", arguments=arguments)
    fragments = convert(conv, StreamingChunk(content="", index=3, tool_calls=[delta]))
    deltas = [p["tool_calls"][0] for p in fragments]  # type: ignore[index]
    assert "".join(d["arguments"] for d in deltas) == arguments
    assert deltas[0]["id"] == "c9" and all(d["id"] is None and d["tool_name"] is None for d in deltas[1:])
    for frame in conv.convert(StreamingChunk(content=text), MAIN):
        assert frame_size(frame.data) <= 1024


def test_a_large_result_is_truncated_to_fit() -> None:
    call = weather_call()
    frames = NativeConverter(max_frame_bytes=1024).convert(result_chunk(call, "€" * 100_000), MAIN)
    assert len(frames) == 1 and frame_size(frames[0].data) <= 1024
    assert json.loads(frames[0].data)["tool_call_result"]["result"].endswith("[truncated, 100000 characters in total]")
