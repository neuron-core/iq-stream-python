from __future__ import annotations

import json
from collections.abc import Hashable, Iterable

from haystack.dataclasses import StreamingChunk
from haystack.dataclasses.streaming_chunk import ToolCallDelta

from haystack_integrations.iq_stream import AISDKConverter
from haystack_integrations.iq_stream.converters.base import frame_size
from tests.checks import verify_ai_sdk
from tests.fakes import RESULTS_DONE, result_chunk, text_turn, tool_turn, weather_call

MAIN: Hashable = ("main", None)


def converter(**options: object) -> AISDKConverter:
    return AISDKConverter(message_id="run-1", **options)  # pyright: ignore[reportArgumentType]


def run(conv: AISDKConverter, chunks: Iterable[StreamingChunk], *, source: Hashable = MAIN) -> list[str]:
    frames = [*conv.start()]
    for chunk in chunks:
        frames += conv.convert(chunk, source)
    frames += conv.finish()
    return [frame.data for frame in frames]


def types(data: list[str]) -> list[str]:
    return [json.loads(item)["type"] for item in data]


def test_a_step_holds_the_model_call_and_its_tool_results() -> None:
    call = weather_call()
    first = tool_turn([call], text=["Let ", "me ", "check."], reasoning=["Weather ", "asked."])
    second = text_turn(["Sunny."])
    data = run(converter(), [*first.chunks, result_chunk(call), RESULTS_DONE, *second.chunks])
    assert types(data) == [
        "start",
        "start-step",
        "reasoning-start",
        "reasoning-delta",
        "reasoning-delta",
        "reasoning-end",
        "text-start",
        *["text-delta"] * 3,
        "text-end",
        "tool-input-start",
        "tool-input-delta",
        "tool-input-delta",
        "tool-input-available",
        "tool-output-available",
        "finish-step",
        "start-step",
        "text-start",
        "text-delta",
        "text-end",
        "finish-step",
        "finish",
    ]
    message = verify_ai_sdk(data)
    assert message.texts == ["Let me check.", "Sunny."]
    assert message.reasoning == ["Weather asked."]
    assert message.tools["call_1"]["input"] == {"city": "Paris"}
    assert message.tools["call_1"]["output"] == "Sunny, 21C"
    assert json.loads(data[0]) == {"type": "start", "messageId": "run-1"}
    assert json.loads(data[-1]) == {"type": "finish", "finishReason": "stop"}


def test_tool_errors_are_output_errors_masked_by_default() -> None:
    call = weather_call()
    chunks = [*tool_turn([call]).chunks, result_chunk(call, "boom: secret", error=True), RESULTS_DONE]
    assert verify_ai_sdk(run(converter(), chunks)).tools["call_1"]["error"] == "An error occurred."
    assert verify_ai_sdk(run(converter(expose_errors=True), chunks)).tools["call_1"]["error"] == "boom: secret"


def test_error_and_cancel_end_the_message() -> None:
    conv = converter()
    data = [f.data for f in [*conv.start(), *conv.convert(StreamingChunk(content="Hi"), MAIN)]]
    data += [f.data for f in conv.error(ValueError("secret"))]
    assert verify_ai_sdk(data).terminal == {"type": "error", "errorText": "An error occurred."}

    conv = converter()
    data = [f.data for f in [*conv.start(), *conv.convert(StreamingChunk(content="Hi"), MAIN), *conv.cancelled()]]
    assert verify_ai_sdk(data).terminal == {"type": "abort", "reason": "cancelled"}


def test_interleaved_sources_never_finish_a_step_under_open_text() -> None:
    a, b = ("a", None), ("b", None)
    one, two = text_turn(["A1 ", "A2"]), text_turn(["B1 ", "B2", " B3"])
    conv = converter()
    data = [f.data for f in conv.start()]
    chunks_b = iter(two.chunks)
    for chunk in one.chunks:
        data += [f.data for f in conv.convert(chunk, a)]
        data += [f.data for f in conv.convert(next(chunks_b), b)]
    for chunk in chunks_b:
        data += [f.data for f in conv.convert(chunk, b)]
    data += [f.data for f in conv.finish()]
    message = verify_ai_sdk(data)
    assert sorted(message.texts) == ["A1 A2", "B1 B2 B3"]
    assert message.steps == 1


def test_an_oversize_input_is_split_and_its_available_chunk_left_out() -> None:
    arguments = json.dumps({"text": "€" * 3_000})
    delta = ToolCallDelta(index=0, tool_name="write", id="c1", arguments=arguments)
    chunks = [
        StreamingChunk(content="", index=0, tool_calls=[delta]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    data = run(converter(max_frame_bytes=1024), chunks)
    assert all(frame_size(item) <= 1024 for item in data)
    message = verify_ai_sdk(data)
    assert "".join(message.tools["c1"]["deltas"]) == arguments
    assert "tool-input-available" not in types(data)


def test_a_large_output_is_truncated_with_a_marker() -> None:
    call = weather_call()
    data = run(converter(max_frame_bytes=1024), [*tool_turn([call]).chunks, result_chunk(call, "z" * 50_000)])
    assert all(frame_size(item) <= 1024 for item in data)
    assert verify_ai_sdk(data).tools["call_1"]["output"].endswith("[truncated, 50000 characters in total]")


def test_custom_events_are_data_parts() -> None:
    assert json.loads(converter().custom("confirm", {"id": 1})[0].data) == {"type": "data-confirm", "data": {"id": 1}}


def test_finish_reason_is_the_last_model_message_s() -> None:
    data = run(converter(), text_turn(["cut"], finish="length").chunks)
    assert json.loads(data[-1]) == {"type": "finish", "finishReason": "length"}
