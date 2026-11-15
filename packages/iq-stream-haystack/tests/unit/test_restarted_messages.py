"""A model message left open by a generator that failed mid-answer, and the next model call from the same caller:
Haystack's FallbackChatGenerator moves to its next generator with the same callback. The page must show the two
apart, as the saved conversation does, while ordinary messages of every provider shape stay whole."""

from __future__ import annotations

import json
from typing import Any

from haystack.dataclasses import ChatMessage, ReasoningContent, StreamingChunk, ToolCall
from haystack.dataclasses.streaming_chunk import ToolCallDelta
from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import AGUIConverter, AISDKConverter, IQStream
from tests.checks import verify_agui, verify_ai_sdk
from tests.fakes import fallback_agent, tool_turn

QUESTION = [ChatMessage.from_user("Weather in Paris?")]
MAIN = ("main", None)


def published(protocol: str, kind: str) -> tuple[list[str], dict[str, Any]]:
    publisher = RecordingPublisher()
    with IQStream(publisher, protocol=protocol) as stream:  # pyright: ignore[reportArgumentType]
        result = fallback_agent(kind).run(QUESTION, streaming_callback=stream.callback)
    assert publisher.outcome == "end" and not stream.failed
    return publisher.data, result


def test_ag_ui_a_retried_answer_is_a_message_of_its_own() -> None:
    data, result = published("ag-ui", "text")
    assert result["last_message"].text == "It is sunny in Paris."
    assert verify_agui(data).texts == ["The weather in Par", "It is sunny in Paris."]


def test_ai_sdk_a_retried_answer_is_a_text_part_of_its_own() -> None:
    data, _ = published("ai-sdk", "text")
    assert verify_ai_sdk(data).texts == ["The weather in Par", "It is sunny in Paris."]


def test_ag_ui_a_retried_tool_call_keeps_its_own_arguments_and_the_cut_one_ends() -> None:
    data, result = published("ag-ui", "tool")
    assert result["last_message"].text == "It is sunny in Paris."
    conversation = verify_agui(data)
    assert [(call_id, args) for call_id, _, args, _ in conversation.calls] == [
        ("call_A", '{"city": "Ro'),
        ("call_B", '{"city": "Paris"}'),
    ]
    assert conversation.results == {"call_B": "Sunny in Paris"}


def test_ai_sdk_a_cut_tool_call_is_an_input_error_and_the_retried_one_runs() -> None:
    data, _ = published("ai-sdk", "tool")
    message = verify_ai_sdk(data)
    assert message.tools["call_A"]["state"] == "output-error"
    assert message.tools["call_A"]["error"] == "The tool call's arguments are not valid JSON."
    assert message.tools["call_B"]["input"] == {"city": "Paris"}
    assert message.tools["call_B"]["output"] == "Sunny in Paris"
    assert message.texts == ["It is sunny in Paris."]


# -- the shapes of ordinary messages, which must stay whole ------------------------------------------------------------


def frames(converter: AGUIConverter | AISDKConverter, chunks: list[StreamingChunk]) -> list[str]:
    out = [*converter.start()]
    for chunk in chunks:
        out += converter.convert(chunk, MAIN)
    return [frame.data for frame in [*out, *converter.finish()]]


def agui() -> AGUIConverter:
    return AGUIConverter(thread_id="thread-1", run_id="run-1")


def test_openai_chat_completions_number_text_and_tool_calls_both_from_zero_in_one_message() -> None:
    chunks = [
        StreamingChunk(content="Let me check.", index=0, start=True),
        StreamingChunk(content="", index=0, start=True, tool_calls=[ToolCallDelta(0, "get_weather", None, "c1")]),
        StreamingChunk(content="", index=0, tool_calls=[ToolCallDelta(0, None, '{"city": "Paris"}')]),
        StreamingChunk(content="", index=1, start=True, tool_calls=[ToolCallDelta(1, "get_weather", None, "c2")]),
        StreamingChunk(content="", index=1, tool_calls=[ToolCallDelta(1, None, '{"city": "Rome"}')]),
        StreamingChunk(content="", finish_reason="tool_calls"),
    ]
    data = frames(agui(), chunks)
    conversation = verify_agui(data)
    assert conversation.texts == ["Let me check."]
    text_id = json.loads(data[1])["messageId"]
    assert [parent for *_, parent in conversation.calls] == [text_id, text_id]  # one model message
    assert [args for _, _, args, _ in conversation.calls] == ['{"city": "Paris"}', '{"city": "Rome"}']


def test_a_provider_that_repeats_the_call_s_id_and_name_on_every_fragment_streams_one_call() -> None:
    fragments = ['{"city"', ': "Paris"}']
    chunks = [
        StreamingChunk(content="", index=0, start=number == 0, tool_calls=[ToolCallDelta(0, "get_weather", part, "c1")])
        for number, part in enumerate(fragments)
    ]
    conversation = verify_agui(frames(agui(), [*chunks, StreamingChunk(content="", finish_reason="tool_calls")]))
    assert [(call_id, args) for call_id, _, args, _ in conversation.calls] == [("c1", '{"city": "Paris"}')]


def test_anthropic_shaped_retry_after_thinking_and_text_starts_a_new_message() -> None:
    # A block opens with an empty chunk (start=True), its deltas follow; the retry numbers its blocks from 0 again.
    cut = [
        StreamingChunk(content="", index=0, start=True, reasoning=ReasoningContent("")),
        StreamingChunk(content="", index=0, reasoning=ReasoningContent("The user asks.")),
        StreamingChunk(content="", index=1, start=True),
        StreamingChunk(content="Paris is", index=1),
    ]
    retry = [
        StreamingChunk(content="", index=0, start=True),
        StreamingChunk(content="It is sunny.", index=0),
        StreamingChunk(content="", finish_reason="stop"),
    ]
    conversation = verify_agui(frames(agui(), cut + retry))
    assert conversation.reasoning == ["The user asks."]
    assert conversation.texts == ["Paris is", "It is sunny."]


def test_id_less_calls_retried_at_the_same_index_stay_apart() -> None:
    # MockChatGenerator streams each call whole, without an id unless one is set.
    first = tool_turn([ToolCall(tool_name="get_weather", arguments={"city": "Rome"})], shape="mock", finish=False)
    second = tool_turn([ToolCall(tool_name="get_weather", arguments={"city": "Paris"})], shape="mock")
    tools = verify_ai_sdk(frames(AISDKConverter(message_id="run-1"), first.chunks + second.chunks)).tools
    assert sorted(tool["input"]["city"] for tool in tools.values()) == ["Paris", "Rome"]
