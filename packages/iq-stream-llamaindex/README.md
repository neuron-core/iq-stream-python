# iq-stream-llamaindex

Publish LlamaIndex runs to browsers through an [IQ-Stream server](https://github.com/neuron-core/iq-stream): a `FunctionAgent`, a `ReActAgent`, a multi-agent `AgentWorkflow`, or any `Workflow`. Your backend starts the run and hands the browser a ticket; the run's events reach the browser as Server-Sent Events while they happen, as [AG-UI](https://docs.ag-ui.com) events or as LlamaIndex's own, and the stream ends with `end` once the answer is saved, or with `fail`.

It is one coroutine, `publish(handler, publisher)`, over the handler that `run()` returns and a publisher of the core client [`iq-stream`](https://pypi.org/project/iq-stream/). It reads `handler.stream_events()`, the interface LlamaIndex documents as stable, and it is that stream's only reader: LlamaIndex allows one. `published()` does the same and hands each event on to your code.

```sh
pip install iq-stream-llamaindex              # any workflow, LlamaIndex's own events (NativeEncoder)
pip install "iq-stream-llamaindex[agents]"    # adds llama-index-core: FunctionAgent, ReActAgent, AgentWorkflow
pip install "iq-stream-llamaindex[agui]"      # adds ag-ui-protocol: AG-UI events (AGUIEncoder)
```

## Publish a run

```python
import asyncio

from iq_stream import IQClient

from iq_stream_llamaindex import AGUIEncoder, RunOutcome, publish

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_APP_KEY and IQ_STREAM_APP_SECRET


async def save_answer(outcome: RunOutcome) -> None:
    print("saving", outcome.status, outcome.result)  # your database: called before the close line


async def main() -> None:
    ticket = iq.ticket()  # hand ticket.to_dict() to the browser
    handler = agent.run(user_msg="What is the weather in Paris?", run_id=ticket.stream_id)
    outcome = await publish(
        handler,
        iq.apublisher(ticket.stream_id),  # its one request to the server opens now
        encoder=AGUIEncoder(thread_id="thread-1"),
        before_close=save_answer,
    )
    print(outcome.status, outcome.frames, outcome.publish_result)


asyncio.run(main())
```

`agent` stands for your agent or workflow. Passing the stream id as `run_id` puts it in LlamaIndex's logs and instrumentation tags (agents take it from llama-index-core 0.14.16 on; 0.14.12 ignored it). LlamaIndex has no synchronous API: `run()` needs a running event loop, so a queue worker wraps the run in `asyncio.run()` as above.

What `publish()` does:

- **It publishes while the run runs.** Each event becomes frames on the publisher as it is read: a text delta is a frame, a tool call is its frames, all on the one request the core client holds open. Keepalives cover a tool that takes minutes, or a run waiting for a human.
- **It saves before it closes.** When the run ends, `publish()` awaits its result, calls `before_close(outcome)`, and only then sends the terminal frames (`RUN_FINISHED`, `RUN_ERROR`) and the close line. A browser that arrives after the close receives `iq:gone` and loads the answer from you, so it must be there.
- **It returns how the run ended.** A failed run is returned, not raised: `publish()` usually runs in a task of its own, where an exception would only be logged. `raise_run_errors=True` raises it after the close.
- **Publishing never breaks the run.** A publisher that gives up (the server refused, the stream grew too large) stops the encoding; the run is still read to its end, `before_close` still runs, and `outcome.published` is `False`.
- **It never hangs.** A run killed with the deprecated `handler.cancel()` publishes no terminal event; `publish()` notices that the run is done and ends the stream a second later.

## In a FastAPI app

The route starts the run, answers with the ticket at once, and leaves the run and its stream to a task:

```python
import asyncio

from fastapi import FastAPI
from iq_stream import IQClient
from pydantic import BaseModel

from iq_stream_llamaindex import AGUIEncoder, RunOutcome, publish

app = FastAPI()
iq = IQClient()
running: set[asyncio.Task[RunOutcome]] = set()


class ChatRequest(BaseModel):
    thread_id: str
    message: str


@app.post("/chat")
async def chat(request: ChatRequest) -> dict[str, str | int]:
    ticket = iq.ticket()
    handler = agent.run(user_msg=request.message, run_id=ticket.stream_id)
    publisher = iq.apublisher(ticket.stream_id)
    task = asyncio.create_task(
        publish(handler, publisher, encoder=AGUIEncoder(thread_id=request.thread_id), before_close=save_answer)
    )
    running.add(task)  # the loop keeps tasks weakly
    task.add_done_callback(running.discard)
    return ticket.to_dict()  # streamId, token, url, expiresAt: what the browser subscribes with
```

`save_answer(outcome)` stands for your persistence. The repository's [`examples/fastapi_agui/app.py`](https://github.com/neuron-core/iq-stream-python/tree/main/packages/iq-stream-llamaindex/examples/fastapi_agui/app.py) adds the two routes a chat needs beside this one: answering a human-input request and stopping a run, each checking the ticket's token with `iq.verify()`. Its tests serve it with uvicorn and drive it over HTTP.

## Reading the events too

`stream_events()` has one reader. Code that still needs the events, to answer a human-input request or to log, reads them from `published()`, which yields each event after publishing it, except the terminal one, which it yields before its frames: those go out when the block is left, after `before_close`, so nothing tells the browser the run ended before your code has seen it:

```python
import asyncio

from iq_stream import IQClient
from workflows.events import HumanResponseEvent, InputRequiredEvent

from iq_stream_llamaindex import AGUIEncoder, published

iq = IQClient()


async def run_with_approval() -> None:
    ticket = iq.ticket()
    handler = approval_agent.run(user_msg="Book the trip", run_id=ticket.stream_id)
    encoder = AGUIEncoder(thread_id="thread-1")
    async with published(handler, iq.apublisher(ticket.stream_id), encoder=encoder) as events:
        async for event in events:
            if isinstance(event, InputRequiredEvent):  # published already, as a CUSTOM "input_required" event
                answer = await answer_from_the_browser(event)
                handler.ctx.send_event(HumanResponseEvent(response=answer))
    print(events.outcome.status)


asyncio.run(run_with_approval())
```

`approval_agent` stands for an agent whose tool calls `ctx.wait_for_event(HumanResponseEvent, waiter_event=InputRequiredEvent(...))`, and `answer_from_the_browser()` for your endpoint that receives the answer. The stream stays open while the run waits, on keepalives, and the same stream goes on once it is answered; keep the handler reachable by stream id for that, as the example app does. A `ToolCall` that the wait replays is announced once.

Breaking out of the loop early does not cut the browser's stream short: leaving the block reads and publishes the rest of the run. Leaving it with an exception stops the run (`handler.cancel_run()`) and closes the stream with `fail`. `publish(..., on_event=...)` is the same with a callback. An `AgentOutput` is yielded, and published, once the next event has arrived: llama-index-core changes it after writing it to the stream (a `ReActAgent` strips its `Answer:` scaffolding, the last one gains the run's tool calls), and it is published as finalized, as your result has it.

## What the browser needs

**`AGUIEncoder`** publishes one AG-UI event per frame, unnamed, as AG-UI's own SSE transport sends them. In the browser, `StreamAgent` from [`@neuron-core/iq-stream/ag-ui`](https://www.npmjs.com/package/@neuron-core/iq-stream) is an `AbstractAgent` of `@ag-ui/client`, so CopilotKit and any AG-UI client can drive it: its `start(input)` posts the run's input to your `/chat` route and returns the ticket. A failed run arrives as `RUN_ERROR` and a stopped one as `RUN_FINISHED` with the `cancelled` outcome, followed by `iq:fail`; `StreamAgent` reads the first and does not report the second as another error. The tests replay what real subscribers received into `StreamAgent.runAgent()` and check the messages it rebuilds: the text, the tool call with its arguments and result, the outcome.

**`NativeEncoder`** publishes each event under its class name, with JSON data: `RunStarted`, `AgentStream`, `AgentOutput`, `ToolCall`, `ToolCallResult`, `InputRequiredEvent`, your own events, and the terminal `StopEvent`, `WorkflowFailedEvent`, `WorkflowTimedOutEvent`, `WorkflowCancelledEvent` or `RunAborted`. A page listens to those names, with `addEventListener` on an `EventSource` or with `frames(url, { events: [...] })` from `@neuron-core/iq-stream/client`, and parses each frame's data. It suits pages without a chat framework and workflows whose events are your own.

Either way the stream ends with `iq:end` or `iq:fail`, or `iq:gone` for a browser that came after the close and loads the answer from you.

## Event mapping

| Stream event | `NativeEncoder`: event name and data | `AGUIEncoder` |
| --- | --- | --- |
| (before the first) | `RunStarted` `{run_id}` | `RUN_STARTED` |
| `AgentInput`, `AgentSetup` | not published: prompts and history stay on the server | a new agent: `STEP_FINISHED` of the previous one, `STEP_STARTED` |
| `AgentStream` | `AgentStream` `{delta, agent}`, with `thinking_delta` when there is one | `REASONING_*` for a thinking delta, `TEXT_MESSAGE_START` / `CONTENT` for a text delta |
| `AgentOutput` | `AgentOutput` `{agent, content, tool_calls}`; the last one's `tool_calls` are the run's, as in the result | `TEXT_MESSAGE_END`; the whole text, for an agent built with `streaming=False` |
| `ToolCall` (a replay left out) | `ToolCall` `{tool_id, tool_name, tool_kwargs}` | `TOOL_CALL_START`, `TOOL_CALL_ARGS`, `TOOL_CALL_END` |
| `ToolCallResult` | `ToolCallResult` `{tool_id, tool_name, content, is_error, return_direct}` | `TOOL_CALL_RESULT` |
| `AgentStreamStructuredOutput` | `{output}` | `CUSTOM` `structured_output` |
| `InputRequiredEvent` | its fields | `CUSTOM` `input_required` |
| An AG-UI event (from `AGUIChatWorkflow`) | its AG-UI JSON | passed through unchanged |
| Your own event | its fields, undeclared ones included | `CUSTOM` named after its class (`custom_events=False` drops it) |
| `StopEvent` | `{result}` | `STEP_FINISHED`, `RUN_FINISHED`; first the answer as a text message when a `return_direct` tool gave it |
| `WorkflowFailedEvent` | `{step_name, message}` | `RUN_ERROR` code `failed` |
| `WorkflowTimedOutEvent` | `{timeout, active_steps}` | `RUN_ERROR` code `timeout` |
| `WorkflowCancelledEvent` | `{}` | `RUN_FINISHED` with outcome `cancelled` |
| (no terminal event; the answer not saved) | `RunAborted` `{reason}` | `RUN_FINISHED` `cancelled`, or `RUN_ERROR` code `aborted` |

## Which close line

| The run | Terminal frames | Close |
| --- | --- | --- |
| completed | `StopEvent` / `RUN_FINISHED` | `end`, after `before_close` |
| a step failed for good, or it timed out | `WorkflowFailedEvent`, `WorkflowTimedOutEvent` / `RUN_ERROR` | `fail`, after `before_close` |
| cancelled with `handler.cancel_run()` | `WorkflowCancelledEvent` / `RUN_FINISHED` `cancelled` | `fail`, or `end` with `cancelled_close="end"`, after `before_close` |
| waiting for a human | nothing: the stream stays open | none yet |
| the task running `publish()` was cancelled | the encoder's abort frames | `fail`; the run is stopped and the cancellation goes on |
| `before_close` raised | the encoder's abort frames instead of the terminal ones | `fail`, and the exception goes on |
| the run ended, but no terminal event came through its stream | the encoder's abort frames | `fail` |
| the publisher gave up | none | none: the core client already closed the stream with `fail` |

A cancelled run closes with `fail` by default because its answer is incomplete, and a second tab watching the stream should not take it for a finished one; the `cancelled` outcome in the data tells the page that it was a stop.

## Reference

```text
publish(handler, publisher, *, encoder=None, on_event=None, before_close=None, cancelled_close="fail",
        raise_run_errors=False) -> RunOutcome
published(handler, publisher, *, encoder=None, before_close=None, cancelled_close="fail") -> PublishedStream
```

| Parameter | |
| --- | --- |
| `handler` | What `workflow.run(...)` or `agent.run(...)` returned, whose events nothing has read yet. |
| `publisher` | `iq.apublisher(stream_id)`, created on the running loop; any `AsyncPublisherLike`. |
| `encoder` | `NativeEncoder()` by default, `AGUIEncoder(thread_id=...)`, or your own `Encoder`. |
| `on_event` | Called with each event after its frames were sent, the terminal event before its frames (they follow `before_close`); may be async. Its exceptions propagate. |
| `before_close` | Called with the `RunOutcome` before the terminal frames and the close line; may be async. Its exceptions propagate, the stream closed with `fail`. |
| `cancelled_close` | The close line of a cancelled run: `"fail"` or `"end"`. |
| `raise_run_errors` | Raise the run's exception after the close instead of returning it. |

`RunOutcome` has `status` (`completed`, `failed`, `timed_out`, `cancelled`), `run_id`, `result` (what `await handler` returned: an agent's `AgentOutput`), `stop_event`, `error` (what `await handler` raised), `frames` (frames the publisher accepted), `published` (false once the publisher gave up) and `publish_result` (the server's answer to the close line, `None` in `before_close`).

`NativeEncoder(*, include=(), exclude=(), max_field_bytes=16_384, expose_errors=False)`: `include` and `exclude` choose events by class; the terminal event always goes out.

`AGUIEncoder(*, thread_id, run_id=None, emit_steps=True, custom_events=True, max_field_bytes=16_384, expose_errors=False)`: `run_id` defaults to the handler's; `emit_steps` publishes a step per agent; `custom_events` publishes your own events as `CUSTOM`.

Both mask the text of the exceptions they encode, a failing tool's included, as `"An error occurred."` unless `expose_errors=True`: an error message can carry what a browser should not see. Events that already are AG-UI events, which `AGUIChatWorkflow` writes itself, pass through as they are: its `MESSAGES_SNAPSHOT` carries each tool message, and for a backend tool that raised, `AGUIChatWorkflow` stores the exception's text as that message, so such a tool should catch its own exceptions and return what the browser may see.

Tool calls: a `ToolCall` that a human-input wait replays (the same id, tool and arguments, with no result yet) is announced once. Some model integrations have no tool-call ids and use the tool's name (Ollama, Gemini at times), so two calls can share an id: `NativeEncoder` keeps LlamaIndex's ids, and `AGUIEncoder` publishes the second call of a run under `<id>#2`, the third under `<id>#3`, with its result under the same id.

An encoder serves one run at a time: `start()` begins a run and drops what the encoder kept about the previous one, so one instance can serve runs one after another, not two at once. An `Encoder` of your own has four methods, each returning frames, `(event_name_or_None, data)`: `start(run_id)`, `encode(event)` for every event the terminal one included, `abort(error)` for a stream that ends without one, and `fit(frame, fits)`, what to send instead of a frame over the line limit (several frames, a shorter one, or none).

## Sizes

- **Large fields are cut.** Tool arguments, tool results and final answers longer than `max_field_bytes` (16 KiB) are cut, with `"truncated": true` in native frames and an ellipsis in AG-UI ones. Tool arguments stay JSON.
- **Frames fit the publisher's limit.** A frame over the client's `max_frame_bytes` (65,536 by default; set it to your app's limit if the server has a lower one) is handed to the encoder's `fit()`: a long delta or tool-argument text is sent as several frames, which a page concatenates as it would the original; a tool result is cut further. What cannot be made to fit, such as an `AGUIChatWorkflow` `MESSAGES_SNAPSHOT` of a long conversation, is dropped with a warning on the `iq_stream_llamaindex` logger that names the event, never its data.
- **Text deltas are small.** The native `AgentStream` frame carries the delta, not the cumulative `response` LlamaIndex repeats in every event, so a stream grows with the answer, not with its square.
- **Every frame is JSON a browser parses.** A non-finite float (`NaN`, `Infinity`) is published as `null`, and a lone surrogate (half of a pair split across two streamed chunks, text decoded with `surrogateescape`) as U+FFFD, which is what the server would make of it.

## Limits worth knowing

- One reader: a handler whose events something else read, or whose stream a generator of `stream_events()` that was not closed still holds, is refused with an error that says so, and the stream closes with `fail`.
- The server's stream limits (10 MiB and 30 minutes by default) apply to a run waiting for a human too; `ctx.wait_for_event()` has its own timeout.
- Tool arguments are published whole at `ToolCall`: LlamaIndex parses them before the step starts, so there is no argument stream to relay.
- A `ReActAgent`'s text includes its `Thought:` and `Action:` scaffolding, as LlamaIndex streams it, and as its tool steps' `AgentOutput` carries it with `streaming=False`; its last answer is published without `Answer:`, as LlamaIndex saves it, unless it was streamed.
- A workflow that a tool starts on its own (an agent used as a tool) has its own handler; its events are in the stream only if the tool forwards them with `ctx.write_event_to_stream()`.
- Reasoning deltas appear when the model integration puts `thinking_delta` in its chunks, as the Anthropic one does.

## Not covered

- A tap at the runtime for when another component already reads the handler's stream (LlamaIndex's AG-UI router, `WorkflowServer`): not in this release.
- AI SDK UI message chunks for `useChat`: no encoder yet; `AGUIEncoder` and `NativeEncoder` are the two formats.
- Durable runtimes, which replay steps and so events, and `WorkflowServer`.

## Testing your own code

`AsyncRecordingPublisher` from `iq_stream.testing` records frames and the close line in memory:

```python
from iq_stream.testing import AsyncRecordingPublisher

from iq_stream_llamaindex import publish


async def test_the_agent_answers_and_the_stream_ends() -> None:
    publisher = AsyncRecordingPublisher()
    outcome = await publish(agent.run(user_msg="Weather in Paris?"), publisher)
    assert outcome.status == "completed"
    assert publisher.outcome == "end"
    assert publisher.frames[-1].event == "StopEvent"
```

## Requirements

- Python 3.10 to 3.13.
- `llama-index-workflows` 2.14 or later in 2.x; `llama-index-core` 0.14 for the agents; `ag-ui-protocol` 1.x for AG-UI.
- `iq-stream` 0.1 and an IQ-Stream server 1.x.

## License

MIT
