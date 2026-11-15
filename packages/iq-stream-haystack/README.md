# iq-stream-haystack

Publish a [Haystack](https://haystack.deepset.ai) agent's or pipeline's run to browsers through an [IQ-Stream server](https://github.com/neuron-core/iq-stream). `IQStream` is a context manager around the run that gives Haystack its streaming callback. Every `StreamingChunk` the run produces becomes frames, AG-UI events by default, AI SDK UI message chunks or Haystack's own chunks if you prefer, and the core client [`iq-stream`](https://pypi.org/project/iq-stream/) publishes them over one request. The stream closes with `end` when the run finished and its answer was saved, and with `fail` when it did not. The browser reads it with [`@neuron-core/iq-stream`](https://www.npmjs.com/package/@neuron-core/iq-stream).

```sh
pip install iq-stream-haystack
pip install "iq-stream-haystack[hayhooks]"   # with a Hayhooks release this package was tested with
```

It imports as `haystack_integrations.iq_stream`, the namespace of Haystack integrations, so a pipeline built with its module-level callback loads from YAML without configuration.

## Publish a run

The process that answers the web request creates an `IQStream` to get a stream id and hands the browser its ticket. The process that runs the agent enters a second `IQStream` with the same id around the run:

```python
from concurrent.futures import ThreadPoolExecutor

from iq_stream import IQClient

from haystack_integrations.iq_stream import IQStream

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_PUBLIC_URL, IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET
jobs = ThreadPoolExecutor()


def start(chat_id: str) -> dict[str, str | int]:
    """The web request: queues the run and answers with the ticket at once."""
    stream = IQStream(iq)  # opens nothing: it makes the stream id
    jobs.submit(answer, chat_id, stream.stream_id)
    return stream.ticket().to_dict()  # streamId, token, url, expiresAt: what the browser subscribes with


def answer(chat_id: str, stream_id: str) -> None:
    """Where the agent runs, which is where the publisher must be."""
    with IQStream(iq, stream_id, thread_id=chat_id) as stream:  # the request to the server opens now
        result = agent.run(load_messages(chat_id), streaming_callback=stream.callback)
        save_messages(chat_id, result["messages"])  # before the close line: a late browser loads it from you
    # end, or fail if the block raised


ticket = start("chat-1")
jobs.shutdown(wait=True)
```

`agent` stands for your `Agent`, `load_messages()` and `save_messages()` for your persistence. Entering the block opens the publisher and sends the protocol's start event (`RUN_STARTED`). Leaving it sends the final event (`RUN_FINISHED`, or `RUN_ERROR` when the block raised) and closes the stream; the run's exception goes on unchanged. A subscriber waits 30 seconds for its publisher: when the job may wait longer in your queue, call `iq.reserve(stream_id)` where you hand out the ticket.

In asyncio, enter the block with `async with` and pass `acallback` (or `callback`, see below):

```python
import asyncio

from iq_stream import IQClient

from haystack_integrations.iq_stream import IQStream

iq = IQClient()


async def answer(chat_id: str, stream: IQStream) -> None:
    async with stream:  # closed even when this task is cancelled
        result = await agent.run_async(load_messages(chat_id), streaming_callback=stream.acallback)
        await asave_messages(chat_id, result["messages"])


async def main() -> None:
    stream = IQStream(iq, thread_id="chat-1")
    ticket = stream.ticket()  # hand it to the browser
    await answer("chat-1", stream)


asyncio.run(main())
```

`callback` and `acallback` are bound methods, as Haystack requires: it awaits only a coroutine function, and it deep-copies a pipeline's inputs, so `IQStream` returns itself from `copy.deepcopy`. `callback` is thread-safe and never blocks: tools, sub-agents and `asyncio.to_thread` workers call it, and it only queues frames for the core's writer thread.

Which one `run_async` takes depends on every chat generator the callback reaches, not only the agent's own: Haystack runs a generator that has no `run_async` in a thread, and there it refuses a coroutine callback with `ValueError: The runtime callback cannot be a coroutine.`, which fails the run. Pass `acallback` only when each of them has `run_async`: the agent's generator, the generators a `FallbackChatGenerator` wraps, and those of sub-agents that get the callback through `tool_streaming_callback_passthrough`. Otherwise pass `callback`, which works in every case; Haystack then logs that a sync callback may block the loop, which this one does not.

## Pipelines

A callback given at run time goes into the component's inputs:

```python
from haystack import Pipeline

from haystack_integrations.iq_stream import IQStream

pipe = Pipeline()
pipe.add_component("agent", agent)

with IQStream(iq) as stream:
    result = pipe.run({"agent": {"messages": load_messages("chat-1"), "streaming_callback": stream.callback}})
    save_messages("chat-1", result["agent"]["messages"])
```

A pipeline that is serialized takes the module-level `iq_streaming_callback` (or `iq_streaming_callback_async`) at construction. It publishes to the `IQStream` whose block is active in the calling context, thread or task, and does nothing outside one:

```python
from haystack import Pipeline
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator

from haystack_integrations.iq_stream import IQStream, iq_streaming_callback

pipe = Pipeline()
model = MockChatGenerator(responses=["It is sunny in Paris."])  # stands for your chat generator
pipe.add_component("agent", Agent(chat_generator=model, streaming_callback=iq_streaming_callback))
yaml = pipe.dumps()  # streaming_callback: haystack_integrations.iq_stream.callbacks.iq_streaming_callback

loaded = Pipeline.loads(yaml)  # no allowed_modules: Haystack trusts the haystack_integrations namespace
with IQStream(iq) as stream:
    loaded.run({"agent": {"messages": load_messages("chat-1")}})
```

A generator streams whenever it has a callback, so with the module-level one it streams outside a block too, where the callback does nothing.

## In Hayhooks, or another server

`haystack_integrations.iq_stream.hayhooks.start_stream` starts an agent or a pipeline in a task of the running loop and returns the ticket at once; `before_close(stream_id, result)` runs inside the stream's block, before its close line, and is where you save the answer. A Hayhooks pipeline wrapper returns it from `run_api_async` (from [`examples/hayhooks/weather_agent/pipeline_wrapper.py`](https://github.com/neuron-core/iq-stream-python/tree/main/packages/iq-stream-haystack/examples/hayhooks/weather_agent/pipeline_wrapper.py), which the tests deploy in a real Hayhooks server):

```python
from hayhooks import BasePipelineWrapper
from haystack.components.agents import Agent
from haystack.components.generators.chat import MockChatGenerator
from haystack.dataclasses import ChatMessage, ToolCall
from haystack.tools import tool
from iq_stream import IQClient

from haystack_integrations.iq_stream.hayhooks import start_stream

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_PUBLIC_URL, IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET
...
class PipelineWrapper(BasePipelineWrapper):
...
    async def run_api_async(
        self,
        question: str,
        thread_id: str | None = None,
        run_id: str | None = None,
        protocol: Literal["ag-ui", "ai-sdk"] = "ag-ui",
    ) -> dict[str, Any]:
        """Starts the agent and returns what the browser needs to subscribe."""
        run_args = {"messages": [ChatMessage.from_user(question)]}
        return await start_stream(
            iq, self.agent, run_args, before_close=save, thread_id=thread_id, run_id=run_id, protocol=protocol
        )
```

`POST /weather_agent/run` with `{"question": ..., "thread_id": ...}` then answers `{"result": {"streamId": ..., "token": ..., "url": ..., "expiresAt": ...}}`: Hayhooks wraps what `run_api_async` returns in `result`, so the page reads the ticket there. `protocol` lets the example serve both bridges below; an application publishes the one its page reads. For a pipeline, `run_args` is its data in nested form and the stream's callback goes to every component that takes a `streaming_callback`, teed with one the data already gives. `start_stream` gives a component `acallback` only when that is safe by the rule above, and `callback` when it cannot tell, as for any agent that passes its callback to its tools (`tool_streaming_callback_passthrough=True`), whether they come at construction or at run time. Keyword arguments beyond `before_close` and `ttl` go to `IQStream` (`stream_id`, `protocol`, `thread_id`, `run_id`, `close_on_cancel`, ...). An exception in the run closes the stream with `fail` and is logged; with a client opened with `on_error="raise"`, a publishing failure after a run that completed is logged as that, not as a failed run. `start_stream_sync` does the same from `run_api`, in a thread. Neither imports Hayhooks, so a FastAPI route can return them just as well. The run belongs to the serving process: it ends with that process, and its subscribers then get `iq:fail`.

In the page, `StreamAgent` from `@neuron-core/iq-stream/ag-ui` posts what the route reads, with AG-UI's `threadId` and `runId`, so that the events carry the client's own ids, and returns the ticket:

```ts
import { StreamAgent } from '@neuron-core/iq-stream/ag-ui';

const agent = new StreamAgent({
  threadId: 'chat-1',
  start: async ({ threadId, runId, messages }) => {
    const response = await fetch('/weather_agent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: messages.at(-1)?.content ?? '', thread_id: threadId, run_id: runId }),
    });
    return (await response.json()).result; // the ticket, which Hayhooks wraps in "result"
  },
});

agent.addMessage({ id: 'user-1', role: 'user', content: 'Weather in Paris?' });
await agent.runAgent();
```

With `protocol="ai-sdk"`, `useChat` reads the stream through `chatTransport` from `@neuron-core/iq-stream/ai-sdk`:

```ts
import { chatTransport } from '@neuron-core/iq-stream/ai-sdk';

const transport = chatTransport({
  start: async ({ chatId, messages }) => {
    const question = messages.at(-1)?.parts.find((part) => part.type === 'text')?.text ?? '';
    const response = await fetch('/weather_agent/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, thread_id: chatId, protocol: 'ai-sdk' }),
    });
    return (await response.json()).result;
  },
});
// const { messages, sendMessage, status } = useChat({ id: chatId, transport });
```

Both snippets run in Node in the integration tests, against the example deployed in a real Hayhooks server and a real IQ-Stream server (Node's `EventSource`, `AbstractChat` of the AI SDK for `useChat`); the React hook itself is not run.

## Human in the loop

Haystack's `ConfirmationHook` asks a strategy before a tool runs, and the strategy blocks until it has an answer. Give it the stream through `hook_context`: it publishes its request with `custom()`, which reaches the page in the stream's own protocol, and the stream stays open on the core's keepalives while it waits. `custom()` never raises into the run; it returns `False` when the frame was not queued, as when the arguments make it too large for a frame:

```python
from typing import Any

from haystack.components.agents import Agent
from haystack.hooks.human_in_the_loop import ConfirmationHook, ToolExecutionDecision

from haystack_integrations.iq_stream import IQStream


class AskTheBrowser:
    def run(self, *, tool_name: str, tool_description: str, tool_params: dict[str, Any], tool_call_id: str | None = None,
            confirmation_strategy_context: dict[str, Any] | None = None) -> ToolExecutionDecision:  # fmt: skip
        stream: IQStream = (confirmation_strategy_context or {})["stream"]
        request = {"toolCallId": tool_call_id, "tool": tool_name, "arguments": tool_params}
        if not stream.custom("tool_confirmation", request):  # too large: the page loads the arguments from you
            stream.custom("tool_confirmation", {**request, "arguments": None})
        approved = wait_for_answer(tool_call_id)  # your endpoint answers it
        feedback = None if approved else "The user declined."
        return ToolExecutionDecision(tool_name, approved, tool_call_id, feedback, tool_params)

    async def run_async(self, **kwargs: Any) -> ToolExecutionDecision:
        return self.run(**kwargs)


hook = ConfirmationHook(confirmation_strategies={"get_weather": AskTheBrowser()})
careful_agent = Agent(chat_generator=chat_generator, tools=[get_weather], hooks={"before_tool": [hook]})
with IQStream(iq) as stream:
    careful_agent.run(load_messages("chat-1"), streaming_callback=stream.callback, hook_context={"stream": stream})
```

The run goes on in the same stream once the strategy returns, and closes with `end` when it completes. The request is an AG-UI `CUSTOM` event, an AI SDK `data-tool_confirmation` part, or `{"type": "custom"}` in Haystack's format; in each, a value JSON cannot hold is written as its `str()`, and a NaN or an infinity as `null`, as `JSON.stringify` writes them. `emit(payload, event=None)` publishes a frame exactly as given instead: an `ag_ui.core` event, a mapping as JSON, or a string. A frame of either kind that cannot be encoded (a mapping with keys that are not strings, a reference cycle) or is too large is dropped and reported to `on_dropped`, and the stream goes on.

## What the browser needs

**AG-UI** (`protocol="ag-ui"`, the default): one AG-UI event per frame, unnamed. `StreamAgent` from `@neuron-core/iq-stream/ag-ui` is an `AbstractAgent` of `@ag-ui/client`, so CopilotKit and any AG-UI client can drive it; its `start(input)` posts the run's input to your endpoint and returns the ticket. A failed run arrives as `RUN_ERROR` and a cancelled one as `RUN_FINISHED` with the `cancelled` outcome, followed by `iq:fail`: `StreamAgent` takes the first as the run's end and does not report the second as another error.

**AI SDK** (`protocol="ai-sdk"`): one UI message chunk per frame, unnamed, for `useChat` with `chatTransport` from `@neuron-core/iq-stream/ai-sdk`. The whole run is one assistant message, a step per model call with its tool results. A failed run ends with an `error` chunk, which is the chat's error; a cancelled one with an `abort` chunk, after which the chat keeps the partial message. No `[DONE]` frame is sent: `chatTransport` parses every frame as JSON and ends at the IQ terminal event.

**Haystack** (`protocol="haystack"`): each chunk as `{"type": "chunk", ...}`, the fields of `StreamingChunk.to_dict()` without `meta` and the providers' `extra`, between `{"type": "start"}` and `{"type": "finish"}`, `{"type": "error", "message": ...}` or `{"type": "cancelled"}`. A page reads it with `frames(url, { parse: JSON.parse })` from `@neuron-core/iq-stream/client`.

The integration tests replay what real subscribers received into `StreamAgent.runAgent()`, into `chatTransport` under the AI SDK's `AbstractChat` (the state machine of `useChat`) and into `frames()`, and check what each rebuilds: the text, the reasoning, the tool calls with their arguments and results, and how the run ended. A browser that comes after the close gets `iq:gone` and loads the saved messages from you; `ChatMessage.to_dict()` serializes them.

## Event mapping

| Haystack | AG-UI | AI SDK |
| --- | --- | --- |
| entering the block | `RUN_STARTED {threadId, runId}` | `start {messageId: runId}` |
| first chunk of a model message | | `start-step` |
| reasoning text | `REASONING_START`, `REASONING_MESSAGE_START`, `REASONING_MESSAGE_CONTENT`; `*_END` when text or a tool call follows | `reasoning-start`, `reasoning-delta`, `reasoning-end` |
| `content` | `TEXT_MESSAGE_START`, `TEXT_MESSAGE_CONTENT`; `TEXT_MESSAGE_END` when a tool call follows | `text-start`, `text-delta`, `text-end` |
| a `ToolCallDelta` with a new index | `TOOL_CALL_START {toolCallId, toolCallName, parentMessageId}` | `tool-input-start` |
| its `arguments` | `TOOL_CALL_ARGS` | `tool-input-delta` |
| `finish_reason` of a model message | what is open ends; `TOOL_CALL_END` per call | what is open ends; `tool-input-available` per call; `finish-step` unless tool results follow |
| `tool_call_result` | `TOOL_CALL_RESULT {toolCallId, content, role: "tool"}` | `tool-output-available`, or `tool-output-error` |
| `finish_reason == "tool_call_results"` | | `finish-step` |
| leaving the block | `RUN_FINISHED` | `finish {finishReason}` |
| leaving with an exception | `RUN_ERROR {message, code: "error"}` | `error {errorText}` |
| leaving with `CancelledError` | `RUN_FINISHED {outcome: {type: "cancelled"}}` | `abort {reason: "cancelled"}` |
| `custom(name, value)` | `CUSTOM {name, value}` | `data-<name> {data}` |

A model message that a failing generator leaves open, as when `FallbackChatGenerator` moves to its next generator mid-answer, is closed when the next one starts streaming from the same thread or task (its blocks start again), or else when the agent streams the tool results of that step, so the page shows the cut-off part and the new answer apart. A tool call cut off that way, or by the `length` limit, has arguments that are not JSON and gets no result, since Haystack runs no such call: AG-UI ends it, and the AI SDK marks it with `tool-input-error`. Empty deltas publish nothing, so Anthropic's empty reasoning and signature chunks are left out; `ReasoningContent.extra`, `ToolCallDelta.extra` and `chunk.meta` are never published. Model messages are told apart by the thread and task that call back, since one model response always comes from one of them while sub-agents interleave. A tool call without a provider id gets one, and its result is matched by tool name and arguments. A result for a call the browser never saw is preceded by that call. Tool results that are content parts become their text, with `[ImageContent]` or `[FileContent]` in place of the rest.

## Which close line

| The run | Final frame | Close |
| --- | --- | --- |
| completed, its answer saved in the block | `RUN_FINISHED` / `finish` | `end` |
| raised, a tool with `raise_on_tool_invocation_failure=True` included, or the save raised | `RUN_ERROR` / `error` | `fail`, and the exception goes on |
| cancelled (`task.cancel()`) | `RUN_FINISHED` `cancelled` / `abort` | `fail`, or `end` with `close_on_cancel="end"`; the cancellation goes on |
| waiting for a human in a confirmation strategy | nothing: the stream stays open | none yet |
| reached `max_agent_steps`, or its `exit_reason` is `length` or `content_filter` | `RUN_FINISHED` / `finish` | `end`: the run returned; raise in the block if you want `fail` |
| a tool failed without `raise_on_tool_invocation_failure` | an error result, then the rest of the run | `end` |
| publishing stopped: the publisher gave up or failed (see below) | none | `fail` |

A cancelled run closes with `fail` by default because its answer is incomplete, and a second tab watching the stream should not take it for a finished one; the `cancelled` outcome in the data tells the page that it was a stop.

## Errors and sizes

Publishing never breaks the run. An exception in Haystack's callback would abort the run, so `callback`, `custom()` and `emit()` catch everything. A chunk that cannot be converted, or a frame of your own that cannot be encoded or is too large, is skipped: `on_dropped` gets an `IQError` for it (code `encode_failed`, with the cause as `__cause__`, or a `FrameTooLarge`), and the stream goes on, closing with `end` when the run completes. By default each kind is logged once per stream as a warning on the `haystack_integrations.iq_stream` logger, with the cause's type and never its text or frame data.

Publishing stops when the core's publisher gives up, as when the server cannot be reached or the stream outgrew its limit, or when the publisher fails in another way (code `publish_failed`). `on_dropped` gets that `IQError` once, as soon as the adapter sees it, later chunks are dropped without work, the run is read to its end, and the stream closes with `fail`. The core logs its own give-up; the run, its result and your save are unaffected, and `stream.failed` and `stream.error` say what happened.

An application that would rather fail when the publisher gives up opens its client with `IQClient(on_error="raise")`. Nothing raises into the run even then, so that it reads to its end and your save in the block happens; leaving the block raises the publisher's `IQError` instead. When the block raised an exception of its own, that one goes on, and the adapter logs the publisher's error, which the core does not log in this mode. Skipped frames and a `publish_failed` stop never raise.

Error text reaches a browser masked, as "An error occurred.": the message of `RUN_ERROR`, of the AI SDK's `error` and of an error tool result. `expose_errors=True` publishes it.

A frame must fit the publisher's `max_frame_bytes` (64 KiB by default, or the app's own limit), measured as the server measures it, escapes included. Text, reasoning and argument deltas that do not fit are split into several. A tool result is cut to `max_tool_result_bytes` and, if it still does not fit, to the frame, with a marker that gives its full length; the full result is in the saved messages. An AI SDK `tool-input-available` too large for a frame is left out, since the input deltas carried the arguments. `include_tool_results=False` publishes a placeholder for every result, for tools whose output should not reach the browser.

## Reference

```text
IQStream(client, stream_id=None, *, protocol="ag-ui", thread_id=None, run_id=None, include_tool_results=True,
         max_tool_result_bytes=32_768, expose_errors=False, close_on_cancel="fail", close_timeout=10.0, on_dropped=None)
```

| Option | | |
| --- | --- | --- |
| `client` | | An `IQClient`, whose publisher the block opens, or an open synchronous publisher (`iq_stream.PublisherLike`, such as the testing kit's `RecordingPublisher`); an asyncio publisher is refused with `TypeError`, since Haystack calls back from worker threads. |
| `stream_id` | new | The stream; a new id from the client unless given. |
| `protocol` | `"ag-ui"` | `"ag-ui"`, `"ai-sdk"`, `"haystack"`, or a `ChunkConverter` of your own. |
| `thread_id`, `run_id` | the stream id | AG-UI's `threadId` and `runId`; the AI SDK's `messageId` is `run_id`. |
| `include_tool_results` | `True` | `False` publishes `[tool result omitted]` for every result. |
| `max_tool_result_bytes` | `32_768` | UTF-8 bytes of a tool result before it is truncated. |
| `expose_errors` | `False` | Publish error text instead of "An error occurred.". |
| `close_on_cancel` | `"fail"` | The close line of a cancelled run, an `iq_stream.Outcome`. |
| `close_timeout` | `10.0` | Seconds the close waits for the server's answer, after the core's `min_open`; finite and above 0. A close that is not answered in time ends the stream failed, with `close_timeout`. |
| `on_dropped` | log | Called with an `IQError` for each chunk or frame of your own that was skipped (`encode_failed`, or a `FrameTooLarge`), and once with the error that stopped publishing (the core publisher's, or `publish_failed`). |

| Member | |
| --- | --- |
| `callback(chunk)`, `acallback(chunk)` | The streaming callbacks for `run()` and `run_async()`. |
| `tee(other)` | A callback that publishes, then calls `other`, your own; async when `other` is. A callback given at run time replaces the one a component was built with. |
| `custom(name, value)`, `emit(payload, event=None)` | Frames of your own, in order with the run's. They return whether the frame was queued, and never raise. |
| `ticket(ttl=3600)` | The core's `Ticket` for this stream: `stream_id`, `token`, `url`, `expires_at`, and `to_dict()`. |
| `stream_id`, `publisher`, `failed`, `error`, `result` | The stream, its publisher once entered, whether publishing stopped and why, and the server's answer to the close line. |
| `with`, `async with` | One run per `IQStream`. The async form closes off the event loop and is shielded from cancellation. Leaving raises only the publisher's `IQError`, with a client opened with `on_error="raise"`. |

`start_stream(client, target, run_args, *, before_close=None, ttl=3600, **stream_options)` and `start_stream_sync(...)` are in `haystack_integrations.iq_stream.hayhooks`. `iq_streaming_callback`, `iq_streaming_callback_async` and `current_stream()` are module-level. `ChunkConverter` is the protocol a converter follows (`start`, `convert(chunk, source)`, `finish`, `error(exc)`, `cancelled`, `custom(name, value)`, each returning `Frame` objects), and `AGUIConverter`, `AISDKConverter` and `NativeConverter` are the built-in ones.

## Limits worth knowing

- A stream holds at most 10 MiB and lasts at most 30 minutes, unless the app has limits of its own on the server. An agent with the default `max_agent_steps=100` can exceed both: the server then closes the stream, the core's publisher gives up, and the run goes on unpublished. Its subscribers get `iq:fail` and load the result from you.
- Haystack logs a warning when `run_async` gets a sync callback: use `acallback` where every generator it reaches has `run_async` (see above).
- Sub-agents (`AgentTool` with `tool_streaming_callback_passthrough=True`) publish valid events, interleaved and flat: the page cannot tell which sub-agent a message belongs to.
- A tool call that a confirmation strategy rejects gets no result frame: Haystack writes the rejection into the conversation, not the stream. Publish it with `custom()` if the page shows it.
- The core holds the close line until the stream is `min_open` (3 seconds) old, so that a browser that subscribes just after a quick answer still receives it; leaving the block waits for that.
- An `IQStream` given an `IQClient` uses its synchronous publisher in async runs too, since Haystack calls back from worker threads; the close runs in a thread.

## Not covered

Haystack 2.x; OpenAI chat-completion chunks (Hayhooks' `run_chat_completion` answers on its own connection); `Pipeline.stream()`, which needs no adapter; publishing tracing spans or token usage; telling sub-agents apart.

## Testing your own code

`iq_stream.testing.RecordingPublisher` takes the place of a client's publisher, so a unit test needs no server:

```python
import json

from iq_stream.testing import RecordingPublisher

from haystack_integrations.iq_stream import IQStream


def test_the_answer_reaches_the_browser() -> None:
    publisher = RecordingPublisher()
    with IQStream(publisher) as stream:
        agent.run(load_messages("chat-1"), streaming_callback=stream.callback)
    events = [json.loads(data) for data in publisher.data]
    assert events[0]["type"] == "RUN_STARTED" and events[-1]["type"] == "RUN_FINISHED"
    assert publisher.outcome == "end"
```

## Requirements

- Python 3.10 to 3.13, `haystack-ai` 3.x (tested with 3.5.0, with 3.3.0 on Python 3.10, whose last release it is, and the unit tests with 3.0.0), `ag-ui-protocol` 1.x; `hayhooks` 2.3 or later for the extra.
- An IQ-Stream server 1.x.

## License

MIT
