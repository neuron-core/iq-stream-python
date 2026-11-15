from __future__ import annotations

import json
from collections.abc import Hashable, Iterable

import pytest
from haystack.dataclasses import ImageContent, ReasoningContent, StreamingChunk, TextContent, ToolCall
from haystack.dataclasses.streaming_chunk import ToolCallDelta

from haystack_integrations.iq_stream import AGUIConverter
from haystack_integrations.iq_stream.converters.base import frame_size
from tests.checks import verify_agui
from tests.fakes import RESULTS_DONE, result_chunk, text_turn, tool_turn, weather_call

MAIN: Hashable = ("main", None)


def run(
    converter: AGUIConverter, chunks: Iterable[StreamingChunk], *, source: Hashable = MAIN, end: str = "finish"
) -> list[str]:
    frames = [*converter.start()]
    for chunk in chunks:
        frames += converter.convert(chunk, source)
    frames += converter.finish() if end == "finish" else converter.cancelled()
    return [frame.data for frame in frames]


def converter(**options: object) -> AGUIConverter:
    return AGUIConverter(thread_id="thread-1", run_id="run-1", **options)  # pyright: ignore[reportArgumentType]


def types(data: list[str]) -> list[str]:
    return [json.loads(item)["type"] for item in data]


def test_mock_shaped_text_tool_call_and_answer() -> None:
    call = weather_call()
    first = tool_turn([call], text=["Let ", "me ", "check."], shape="mock")
    second = text_turn(["Sunny ", "in ", "Paris."])
    data = run(converter(), [*first.chunks, result_chunk(call), RESULTS_DONE, *second.chunks])
    assert types(data) == [
        "RUN_STARTED",
        "TEXT_MESSAGE_START",
        *["TEXT_MESSAGE_CONTENT"] * 3,
        "TEXT_MESSAGE_END",
        "TOOL_CALL_START",
        "TOOL_CALL_ARGS",
        "TOOL_CALL_END",
        "TOOL_CALL_RESULT",
        "TEXT_MESSAGE_START",
        *["TEXT_MESSAGE_CONTENT"] * 3,
        "TEXT_MESSAGE_END",
        "RUN_FINISHED",
    ]
    conversation = verify_agui(data)
    assert conversation.texts == ["Let me check.", "Sunny in Paris."]
    assert [(cid, name, json.loads(args)) for cid, name, args, _ in conversation.calls] == [
        ("call_1", "get_weather", {"city": "Paris"})
    ]
    assert conversation.results == {"call_1": "Sunny, 21C"}
    first_event = json.loads(data[0])
    assert first_event == {"type": "RUN_STARTED", "threadId": "thread-1", "runId": "run-1"}
    # The tool call belongs to the text message that announced it.
    assert conversation.calls[0][3] == json.loads(data[1])["messageId"]


def test_openai_shaped_fragments_are_one_call_with_its_arguments_in_order() -> None:
    calls = [weather_call("Paris", "call_a"), weather_call("Rome", "call_b")]
    data = run(converter(), tool_turn(calls, shape="openai").chunks)
    conversation = verify_agui(data)
    assert [(cid, json.loads(args)) for cid, _, args, _ in conversation.calls] == [
        ("call_a", {"city": "Paris"}),
        ("call_b", {"city": "Rome"}),
    ]
    assert types(data).count("TOOL_CALL_ARGS") == 4
    # Parallel calls of one model message share their parent, so a reducer puts them in one assistant message.
    assert len({parent for *_, parent in conversation.calls}) == 1


def test_anthropic_empty_reasoning_and_signatures_publish_nothing_and_no_extra_leaks() -> None:
    call = weather_call()
    turn = tool_turn([call], reasoning=["Thinking ", "hard."], shape="anthropic")
    data = run(converter(), [*turn.chunks, result_chunk(call)])
    conversation = verify_agui(data)
    assert conversation.reasoning == ["Thinking hard."]
    joined = "\n".join(data)
    assert "secret-signature" not in joined
    assert "secret-extra" not in joined
    assert "raw" not in joined
    assert types(data)[:3] == ["RUN_STARTED", "REASONING_START", "REASONING_MESSAGE_START"]


def test_a_message_without_finish_reason_is_closed_by_its_tool_result() -> None:
    call = weather_call()
    turn = tool_turn([call], text=["Checking."], finish=False)
    data = run(converter(), [*turn.chunks, result_chunk(call)])
    sequence = types(data)
    assert sequence.index("TOOL_CALL_END") < sequence.index("TOOL_CALL_RESULT")
    verify_agui(data)


def test_finish_closes_everything_still_open_exactly_once() -> None:
    chunks = [
        StreamingChunk(content="", index=0, reasoning=ReasoningContent("hmm")),
        StreamingChunk(content="", index=1, tool_calls=[ToolCallDelta(index=1, tool_name="t", id="c1", arguments="{")]),
        StreamingChunk(content="partial", index=2),
    ]
    data = run(converter(), chunks)
    sequence = types(data)
    for kind in ("REASONING_END", "TEXT_MESSAGE_END", "TOOL_CALL_END"):
        assert sequence.count(kind) == 1, kind
    assert sequence[-1] == "RUN_FINISHED"
    verify_agui(data)


def test_error_closes_and_masks_the_message_unless_errors_are_exposed() -> None:
    masked = converter()
    data = [f.data for f in [*masked.start(), *masked.convert(StreamingChunk(content="Hi"), MAIN)]]
    data += [f.data for f in masked.error(RuntimeError("database password=hunter2"))]
    conversation = verify_agui(data)
    assert conversation.terminal == {"type": "RUN_ERROR", "message": "An error occurred.", "code": "error"}
    assert "hunter2" not in "".join(data)

    exposed = converter(expose_errors=True)
    data = [f.data for f in [*exposed.start(), *exposed.error(RuntimeError("weather service down"))]]
    assert verify_agui(data).terminal["message"] == "RuntimeError: weather service down"


def test_a_cancelled_run_finishes_with_the_cancelled_outcome() -> None:
    data = run(converter(), [StreamingChunk(content="Hi")], end="cancelled")
    conversation = verify_agui(data)
    assert conversation.terminal == {
        "type": "RUN_FINISHED",
        "threadId": "thread-1",
        "runId": "run-1",
        "outcome": {"type": "cancelled"},
    }


def test_id_less_calls_are_matched_by_name_and_arguments() -> None:
    calls = [weather_call("Paris", None), weather_call("Rome", None)]
    turn = tool_turn(calls, shape="mock")
    # Results arrive in the other order: the arguments decide.
    data = run(converter(), [*turn.chunks, result_chunk(calls[1], "Rome: rain"), result_chunk(calls[0], "Paris: sun")])
    conversation = verify_agui(data)
    by_city = {json.loads(args)["city"]: cid for cid, _, args, _ in conversation.calls}
    assert conversation.results == {by_city["Rome"]: "Rome: rain", by_city["Paris"]: "Paris: sun"}
    assert all(cid.startswith("call_") for cid in by_city.values())


def test_a_result_for_a_call_never_streamed_is_preceded_by_that_call() -> None:
    unknown = ToolCall(tool_name="lookup", arguments={"q": "x"}, id="call_unseen")
    idless = ToolCall(tool_name="lookup", arguments={"q": "y"}, id=None)
    data = run(converter(), [result_chunk(unknown, "found x"), result_chunk(idless, "found y")])
    conversation = verify_agui(data)
    assert [(name, json.loads(args)) for _, name, args, _ in conversation.calls] == [
        ("lookup", {"q": "x"}),
        ("lookup", {"q": "y"}),
    ]
    assert sorted(conversation.results.values()) == ["found x", "found y"]
    assert "call_unseen" in conversation.results


def test_error_results_are_masked_unless_errors_are_exposed() -> None:
    call = weather_call()
    chunks = [*tool_turn([call]).chunks, result_chunk(call, "Failed to invoke Tool: secret stack", error=True)]
    assert verify_agui(run(converter(), chunks)).results == {"call_1": "An error occurred."}
    exposed = verify_agui(run(converter(expose_errors=True), chunks))
    assert exposed.results == {"call_1": "Failed to invoke Tool: secret stack"}


def test_content_part_results_become_text_and_results_can_be_omitted() -> None:
    call = weather_call()
    parts = [TextContent("Sunny"), ImageContent(base64_image="aGVsbG8=", mime_type="image/png"), TextContent("21C")]
    chunks = [*tool_turn([call]).chunks, result_chunk(call, parts)]
    assert verify_agui(run(converter(), chunks)).results == {"call_1": "Sunny\n[ImageContent]\n21C"}
    omitted = verify_agui(run(converter(include_tool_results=False), chunks))
    assert omitted.results == {"call_1": "[tool result omitted]"}


def test_interleaved_sources_each_keep_their_own_messages() -> None:
    first, second = ("thread-a", None), ("thread-b", None)
    one = tool_turn([weather_call("Paris", "p")], text=["A1 ", "A2"])
    two = tool_turn([weather_call("Rome", "r")], text=["B1 ", "B2"])
    conv = converter()
    data = [f.data for f in conv.start()]
    for a, b in zip(one.chunks, two.chunks, strict=True):
        data += [f.data for f in conv.convert(a, first)]
        data += [f.data for f in conv.convert(b, second)]
    data += [f.data for f in conv.convert(result_chunk(weather_call("Rome", "r"), "rain"), MAIN)]
    data += [f.data for f in conv.convert(result_chunk(weather_call("Paris", "p"), "sun"), MAIN)]
    data += [f.data for f in conv.finish()]
    conversation = verify_agui(data)
    assert sorted(conversation.texts) == ["A1 A2", "B1 B2"]
    assert conversation.results == {"r": "rain", "p": "sun"}


def test_a_second_text_block_in_one_message_gets_its_own_id() -> None:
    chunks = [
        StreamingChunk(content="Before. ", index=0),
        StreamingChunk(
            content="", index=1, tool_calls=[ToolCallDelta(index=1, tool_name="t", id="c1", arguments="{}")]
        ),
        StreamingChunk(content="After.", index=2),
        StreamingChunk(content="", finish_reason="stop"),
    ]
    conversation = verify_agui(run(converter(), chunks))
    assert conversation.texts == ["Before. ", "After."]


@pytest.mark.parametrize("limit", [1024, 4096])
def test_an_oversize_argument_delta_is_split_and_concatenates_to_the_original(limit: int) -> None:
    arguments = json.dumps({"text": 'é"\n\\ \u2028 ' * 20_000})  # about 200 kB, with escapes and multi-byte text
    delta = ToolCallDelta(index=0, tool_name="write", id="c1", arguments=arguments)
    conv = converter(max_frame_bytes=limit)
    data = run(
        conv,
        [
            StreamingChunk(content="", index=0, tool_calls=[delta]),
            StreamingChunk(content="", finish_reason="tool_calls"),
        ],
    )
    assert all(frame_size(item) <= limit for item in data)
    conversation = verify_agui(data)
    assert conversation.calls[0][2] == arguments
    assert types(data).count("TOOL_CALL_ARGS") > 1


def test_an_oversize_text_delta_is_split() -> None:
    text = 'Ünïcödé "quoted"\n' * 5_000
    data = run(
        converter(max_frame_bytes=1024),
        [StreamingChunk(content=text), StreamingChunk(content="", finish_reason="stop")],
    )
    assert all(frame_size(item) <= 1024 for item in data)
    assert verify_agui(data).text == text


def test_a_large_tool_result_is_truncated_with_a_marker_that_fits() -> None:
    call = weather_call()
    result = ('x"\n€' * 250_000)[:1_000_000]  # 1,000,000 characters with escapes and multi-byte ones
    data = run(converter(max_frame_bytes=1024), [*tool_turn([call]).chunks, result_chunk(call, result)])
    assert all(frame_size(item) <= 1024 for item in data)
    content = verify_agui(data).results["call_1"]
    assert content.endswith("\n[truncated, 1000000 characters in total]")
    assert result.startswith(content.split("\n[truncated")[0])


def test_max_tool_result_bytes_caps_a_result_that_would_fit_the_frame() -> None:
    call = weather_call()
    data = run(converter(max_tool_result_bytes=100), [*tool_turn([call]).chunks, result_chunk(call, "y" * 5_000)])
    content = verify_agui(data).results["call_1"]
    assert content == "y" * 100 + "\n[truncated, 5000 characters in total]"


def test_a_custom_event_is_a_custom_frame() -> None:
    frames = converter().custom("confirm", {"tool": "delete_file", "path": "/tmp/x"})
    assert json.loads(frames[0].data) == {
        "type": "CUSTOM",
        "name": "confirm",
        "value": {"tool": "delete_file", "path": "/tmp/x"},
    }
