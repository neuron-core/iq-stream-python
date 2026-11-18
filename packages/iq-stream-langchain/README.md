# iq-stream-langchain

Publishes a [LangGraph](https://github.com/langchain-ai/langgraph) run, or a LangChain agent made with `create_agent`, to an [IQ-Stream server](https://github.com/neuron-core/iq-stream), so that browsers read the answer over Server-Sent Events and a reloaded page picks it up where it is. It is a thin layer over [`iq-stream`](https://pypi.org/project/iq-stream/), which holds the connection to the server.

The adapter wraps the iterator of `graph.stream()` / `graph.astream()`, which carries everything the run emits (tokens, tool calls and results, node updates, custom events, sub-graphs, interrupts) in the order LangGraph itself gives them. It is not a callback handler and not middleware: nothing is patched, and a slow or broken publisher cannot delay or fail the run. Each frame is one LangGraph stream part as JSON, the format LangGraph Server sends; AG-UI events are a second format, relayed from [`ag-ui-langgraph`](https://pypi.org/project/ag-ui-langgraph/).

```sh
pip install iq-stream-langchain
pip install "iq-stream-langchain[ag-ui]"   # the AG-UI format, through ag-ui-langgraph and FastAPI
```

## Publish a run

`agent` is a `create_agent` agent or any compiled graph. The publisher's `with` block closes the stream; `publish_run` relays the run to its end and returns the final state, which you save inside the block, before the close.

```python
from langchain.agents import create_agent
from langgraph.checkpoint.memory import InMemorySaver

from iq_stream import IQClient
from iq_stream_langchain import publish_run

iq = IQClient()  # IQ_STREAM_URL, IQ_STREAM_APP_KEY, IQ_STREAM_APP_SECRET
agent = create_agent(model, [get_weather], checkpointer=InMemorySaver())

ticket = iq.ticket()  # hand ticket.to_dict() to the browser
config = {"configurable": {"thread_id": "thread-1"}}
question = {"messages": [{"role": "user", "content": "What is the weather in Turin?"}]}
with iq.publisher(ticket.stream_id) as publisher:  # the request to the server opens now
    final = publish_run(publisher, agent, question, config)
    save_answer(final["messages"])  # before the close: a late browser loads it from you
# the block closes the stream with "end", or with "fail" if the run raised
```

`model` and `get_weather` stand for your chat model and tools, `save_answer()` for your own persistence. With a checkpointer, the thread is saved when the stream of parts ends, so the close line already follows it.

Code that already iterates the stream changes one line: `relay()` yields every part `graph.stream(..., subgraphs=True, version="v2")` yields, and publishes each one on the way.

```python
from iq_stream import IQClient
from iq_stream_langchain import relay

iq = IQClient()
ticket = iq.ticket()
with iq.publisher(ticket.stream_id) as publisher:
    for part in relay(publisher, agent, question, config):
        if part["type"] == "messages" and part["data"][0].type == "AIMessageChunk":
            print(part["data"][0].content, end="")
```

### In a FastAPI app

`arelay()` and `apublish_run()` are the same over `graph.astream()`. The route answers with the ticket at once; the run is published while the page reads it. The same route answers an interrupt, as a new run on a new stream. Three more routes give a reloaded page the run in flight, cancel a run, and load a thread whose stream is over. (Check in each route that the thread is the user's, as your app does for its other data.)

```python
import asyncio
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Response
from langgraph.types import Command
from pydantic import BaseModel

from iq_stream import AsyncPublisher, IQClient
from iq_stream_langchain import apublish_run

app = FastAPI()
iq = IQClient()


@dataclass
class Run:
    stream_id: str
    task: asyncio.Task[None]


running: dict[str, Run] = {}  # thread id -> the run in flight


class ChatIn(BaseModel):
    thread_id: str
    message: str | None = None  # a question, or
    resume: Any = None  # the answer to the interrupt the thread waits on


class StopIn(BaseModel):
    token: str  # the ticket's: it proves the caller holds the stream


@app.post("/chat")
async def chat(body: ChatIn) -> dict[str, str | int]:
    ticket = iq.ticket()
    publisher = iq.apublisher(ticket.stream_id)  # the request to the IQ server opens now
    run = running[body.thread_id] = Run(ticket.stream_id, asyncio.create_task(run_agent(publisher, body)))

    def finished(_: asyncio.Task[None]) -> None:
        if running.get(body.thread_id) is run:  # not a newer run of the thread
            del running[body.thread_id]

    run.task.add_done_callback(finished)
    return ticket.to_dict()  # streamId, token, url, expiresAt


@app.get("/chat/{thread_id}/stream")
async def in_flight(thread_id: str) -> Any:
    """A reloaded page's ticket for the run in flight: the server replays its stream from the first frame."""
    run = running.get(thread_id)
    return iq.ticket(run.stream_id).to_dict() if run else Response(status_code=204)


@app.post("/chat/{stream_id}/stop", status_code=204)
async def stop(stream_id: str, body: StopIn) -> None:
    if not iq.verify("sub", body.token, stream_id):
        raise HTTPException(403, "Not your stream.")
    for run in list(running.values()):
        if run.stream_id == stream_id:
            run.task.cancel()  # the page gets the Cancelled part, then iq:fail


@app.get("/chat/{thread_id}")
async def history(thread_id: str) -> list[dict[str, str]]:
    """What a page loads after iq:gone: the thread as the checkpointer saved it."""
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    return [{"type": message.type, "content": message.text} for message in state.values.get("messages", [])]


async def run_agent(publisher: AsyncPublisher, body: ChatIn) -> None:
    config = {"configurable": {"thread_id": body.thread_id}}
    if body.resume is not None:
        graph_input: Any = Command(resume=body.resume)
    else:
        graph_input = {"messages": [{"role": "user", "content": body.message}]}
    async with publisher:  # "end"; "fail" on an exception or a cancellation
        await apublish_run(publisher, agent, graph_input, config)
```

A graph whose async nodes block the event loop (a synchronous SDK called in an `async def` node) starves an `AsyncPublisher`'s keepalive. Give such a run a synchronous `Publisher`, whose keepalive has a thread of its own, and close it from a thread too: its `close()` waits for the server, which would stall every other task of the app.

```python
import asyncio

from iq_stream import IQClient
from iq_stream_langchain import apublish_run

iq = IQClient()


async def answer(question: dict[str, list[dict[str, str]]]) -> None:
    ticket = iq.ticket()
    publisher = iq.publisher(ticket.stream_id)
    try:
        final = await apublish_run(publisher, agent, question, config)
        save_answer(final["messages"])
    except BaseException:
        await asyncio.to_thread(publisher.close, "fail")
        raise
    await asyncio.to_thread(publisher.close, "end")


asyncio.run(answer(question))
```

Async code that leaves a loop over `arelay()` early wraps it in `contextlib.aclosing()`, so that the stream is closed as a cancelled run at once. Without it Python finalizes the generator later, after the block has closed the stream with `end`.

```python
import asyncio
import contextlib

from iq_stream import IQClient
from iq_stream_langchain import arelay

iq = IQClient()


async def first_words(question: dict[str, list[dict[str, str]]]) -> str:
    ticket = iq.ticket()
    text = ""
    async with iq.apublisher(ticket.stream_id) as publisher:
        async with contextlib.aclosing(arelay(publisher, agent, question, config)) as parts:
            async for part in parts:
                if part["type"] == "messages":
                    text += part["data"][0].text
                    if len(text) > 10:
                        break  # the Cancelled part, then the close line "fail"
    return text


print(asyncio.run(first_words(question)))
```

## What goes over the wire

One frame per stream part, without an SSE event name, its data a JSON object: `{"type": <mode>, "ns": [...], "data": ...}`, with messages and other pydantic objects as `model_dump()` gives them, as LangGraph Server sends them. `ns` is empty for the root graph and names the sub-graph otherwise. Three parts are the adapter's own:

| Part | When |
| --- | --- |
| `{"type": "metadata", "ns": [], "data": {"run_id", "thread_id"}}` | First, before the graph starts, from the config. |
| `{"type": "error", "ns": [], "data": {"error": "<exception type>", "message": "An error occurred."}}` | The run raised. The message is the exception's only with `expose_errors=True`. |
| `{"type": "error", "ns": [], "data": {"error": "Cancelled", "message": "run cancelled"}}` | The run was cancelled. |

By default the modes are `messages`, `updates`, `custom` and `tools` (`DEFAULT_STREAM_MODE`). `values` is left out because each part repeats the whole state, which grows with the thread. A `messages` part keeps the metadata keys the browser clients read (`SLIM_METADATA`), about 430 bytes a token instead of 730.

## In the browser

The frames are LangGraph's own parts, so the page converts them with LangGraph's and the AI SDK's own code. `@neuron-core/iq-stream` has no bridge for this format yet; these functions are the whole of it, on top of its `frames()` and `chatTransport`:

```js
import { frames } from '@neuron-core/iq-stream/client';
import { toUIMessageStream } from '@ai-sdk/langchain';

/** The frames of a stream, as the LangGraph parts iq-stream-langchain published: { type, ns, data }. */
export const langGraphParts = (url, options = {}) => frames(url, { ...options, parse: JSON.parse });

/** For `useStream({ transport })` of @langchain/langgraph-sdk/react: each part as LangGraph Server names it. */
export async function* useStreamEvents(parts) {
  for await (const part of parts) {
    const suffix = part.ns.length ? `|${part.ns.join('|')}` : '';
    yield { event: part.type + suffix, data: part.data };
    // The hook reads interrupts from `values`, which the adapter does not publish by default.
    if (part.type === 'updates' && part.data?.__interrupt__) {
      yield { event: `values${suffix}`, data: { __interrupt__: part.data.__interrupt__ } };
    }
    if (part.type === 'error') return; // the run said how it ended: the iq:fail after it adds nothing
  }
}

/** For `chatTransport({ open })` of @neuron-core/iq-stream/ai-sdk: UI message chunks, converted in the page. */
export function uiMessageChunks(url, options = {}) {
  let cancelled = false;
  let lost; // the subscription failed without the run saying why: iq:gone, publisher_idle, a refusal
  async function* tuples() {
    try {
      for await (const part of langGraphParts(url, options)) {
        if (part.type === 'metadata') continue;
        if (part.type === 'error') {
          cancelled = part.data.error === 'Cancelled';
          throw new Error(part.data.message);
        }
        yield part.ns.length ? [part.ns, part.type, part.data] : [part.type, part.data];
        // @ai-sdk/langchain, too, reads a human-in-the-loop interrupt from `values` only.
        if (part.type === 'updates' && part.data?.__interrupt__) {
          const values = { __interrupt__: part.data.__interrupt__ };
          yield part.ns.length ? [part.ns, 'values', values] : ['values', values];
        }
      }
    } catch (error) {
      if (error?.name === 'IQStreamError') lost = error;
      throw error;
    }
  }
  return toUIMessageStream(tuples()).pipeThrough(
    new TransformStream({
      transform(chunk, controller) {
        if (chunk.type === 'error' && lost) return;
        // A cancelled run is the AI SDK's `abort` chunk, which tells useChat it was stopped, not failed.
        controller.enqueue(chunk.type === 'error' && cancelled ? { type: 'abort' } : chunk);
      },
      flush(controller) {
        if (lost) controller.error(lost); // chatTransport reports it, as for any stream that failed
      },
    }),
  );
}
```

With the AI SDK's `useChat` (`npm install @neuron-core/iq-stream ai @ai-sdk/react @ai-sdk/langchain`), against the routes above:

```ts
import { useChat } from '@ai-sdk/react';
import { chatTransport } from '@neuron-core/iq-stream/ai-sdk';

const transport = chatTransport({
  start: async ({ chatId, messages }) => {
    const message = messages.at(-1)?.parts.find((part) => part.type === 'text')?.text ?? '';
    const response = await fetch('/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ thread_id: chatId, message }),
    });
    return response.json();                                   // the ticket
  },
  resume: async ({ chatId }) => {                             // the run in flight, after a reload
    const response = await fetch(`/chat/${chatId}/stream`);
    return response.status === 204 ? null : response.json();
  },
  stop: (ticket) => void fetch(`/chat/${ticket.streamId}/stop`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ token: ticket.token }),
  }),
  open: (ticket, signal) => uiMessageChunks(ticket.url, { signal }),
});

const { messages, sendMessage, stop, status } = useChat({ id: threadId, transport, resume: true });
// A tool call is a part of type 'dynamic-tool', with toolName, input, output and state; one that waits for a
// human's approval (HumanInTheLoopMiddleware) is in the state 'approval-requested'.
```

`resume: true` is what makes a reloaded page continue the answer: the server replays the stream from its first frame, and `chatTransport` tells `useChat` it is a replay.

With LangGraph's `useStream` (`npm install @neuron-core/iq-stream @langchain/langgraph-sdk`):

```ts
import { useStream } from '@langchain/langgraph-sdk/react';

// stream() gets the input of thread.submit(input), or the command of thread.submit(undefined, { command }).
const transport = {
  async stream({ input, command, config, signal }) {
    const response = await fetch('/chat', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        thread_id: config.configurable.thread_id,
        message: input?.messages?.at(-1)?.content,
        resume: command?.resume,
      }),
      signal,
    });
    const ticket = await response.json();
    return useStreamEvents(langGraphParts(ticket.url, { signal }));
  },
};

const thread = useStream({ threadId, messagesKey: 'messages', transport });
// A run that waits for a human: thread.interrupt holds what it asks, and
// thread.submit(undefined, { command: { resume: answer } }) answers it.
```

A failed run reaches `useStream` as a `StreamError` named after the exception type, and a cancelled one as one named `Cancelled`; `useChat` gets an `error` chunk or an `abort` chunk. When a subscription ends with `iq:gone` (the stream is over and frames this tab missed are gone), load the thread from your backend, as the `/chat/{thread_id}` route above offers. The functions above and the transports ran in Node against what a real subscriber received, through `StreamManager` (the engine of `useStream`, driven as the hook drives it, a resume included), `AbstractChat` (the class behind `useChat`) and `@ai-sdk/langchain` 3.0.172; the React hooks themselves were not run.

## AG-UI

For a front end that speaks [AG-UI](https://docs.ag-ui.com) (CopilotKit, or `StreamAgent` of `@neuron-core/iq-stream/ag-ui`), relay the events of `ag_ui_langgraph.LangGraphAgent`. It owns the run: it needs a checkpointer and calls the graph itself. The run's `RUN_FINISHED` tells the page the answer is complete, so the relay publishes it only after `before_close` has saved the answer.

```python
import asyncio

from ag_ui.core import RunAgentInput, RunFinishedEvent
from ag_ui_langgraph import LangGraphAgent
from fastapi import FastAPI

from iq_stream import AsyncPublisher, IQClient
from iq_stream_langchain.agui import arelay_agui

app = FastAPI()
iq = IQClient()
agui_agent = LangGraphAgent(name="weather", graph=agent, emit_raw_events=False)
running: set[asyncio.Task[None]] = set()


@app.post("/agent")
async def start(run_input: RunAgentInput) -> dict[str, str | int]:
    ticket = iq.ticket()
    task = asyncio.create_task(run(iq.apublisher(ticket.stream_id), run_input))
    running.add(task)
    task.add_done_callback(running.discard)
    return ticket.to_dict()


async def run(publisher: AsyncPublisher, run_input: RunAgentInput) -> None:
    async def save(finished: RunFinishedEvent) -> None:
        state = await agent.aget_state({"configurable": {"thread_id": finished.thread_id}})
        save_answer(state.values["messages"])  # before RUN_FINISHED: a late browser loads it from you

    async with publisher:
        async for _ in arelay_agui(publisher, agui_agent, run_input, before_close=save):
            pass
```

`arelay_agui` clones the agent for each run, as `ag_ui_langgraph`'s own endpoint does (`LangGraphAgent` keeps the state of a run on itself), so one agent serves concurrent requests. Each event is one frame, its JSON as the AG-UI encoder writes it (camelCase), published before it is yielded, except `RUN_FINISHED`: that is yielded when it comes and published once the run's events are over and `before_close(event)` has returned (awaited when it is a coroutine). If `before_close` raises, the page gets a `RUN_ERROR` instead, the stream closes with `fail` and the exception goes on; `StreamAgent` takes `RUN_FINISHED` as the end of the run, and would not report an `iq:fail` that came after it. `ag_ui_langgraph` reports a failed run as a `RUN_ERROR` event instead of raising, so the relay closes the stream with `fail` as soon as it has sent one; its message is replaced by `An error occurred.` unless `expose_errors=True`. A cancelled run ends with `RUN_FINISHED` and the `cancelled` outcome, after the events that close what the client holds open (a message, a tool call, a step, a sub-agent), which AG-UI clients require of a finished run; a loop left at `RUN_FINISHED`, before the save, is a cancelled run too. In the page, `new StreamAgent({ threadId, start: async (input) => (await fetch('/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) })).json() })` reads these frames as they are.

Construct the agent with `emit_raw_events=False`: raw events multiply the size of a stream (69 events and 60 kB against 32 events and 4 kB for one run with a tool call).

## Reference

### `relay(publisher, graph, input, config=None, **options)` and `arelay(...)`

Run `graph.stream(input, config, subgraphs=True, version="v2")` (`astream` for `arelay`), publish each part whose mode is in `stream_mode` as one frame, and yield every part unchanged. `relay` takes a synchronous `Publisher`; `arelay` takes an `AsyncPublisher`, or a `Publisher` for graphs whose nodes block the loop and would starve the asynchronous publisher's keepalive. Both return generators: `close()` / `aclose()` cancel the run.

### `publish_run(publisher, graph, input, config=None, **options)` and `apublish_run(...)`

Relay the run to its end and return the final state of the root graph, its last `values`, which are requested from the graph and not published. A run that waits for a human has its interrupts under `__interrupt__`, as `invoke()` returns them.

| Option | Default | |
| --- | --- | --- |
| `stream_mode` | `DEFAULT_STREAM_MODE` | The modes requested and published: `messages`, `updates`, `custom`, `tools`, and `values`, `tasks`, `checkpoints`, `debug` if you add them. |
| `also_yield` | `()` | Modes requested from the graph and yielded, not published. |
| `metadata` | `"slim"` | The metadata of a `messages` part: `"slim"` (the keys the browser clients read), `"full"`, or a list of keys. |
| `expose_errors` | `False` | Publish the message of a failed run's exception, of a `tool-error` part and of a failed tool's result (a `ToolMessage` with `status="error"`, whose content is the error's text). Otherwise the browser reads `An error occurred.`. |
| `close_on_cancel` | `"fail"` | The close line of a cancelled run, or `"end"` for an application that treats a stopped answer as final. |
| `on_oversize` | `None` | Called with an `Oversize` (`part_type`, `size`, `limit`, `truncated`) for every part over the frame limit. Without it, the first one of a run is logged as a warning on the `iq_stream_langchain` logger. A hook that raises is logged once and never fails the run. |
| `**stream_kwargs` | | Passed to `graph.stream()`: `context`, `durability`, `interrupt_before`, `control`, ... |

`encode_part(part, metadata="slim", expose_errors=False)` and `error_part(error, expose=False)` return the frame data the relay would send, for code of your own. `arelay_agui(publisher, agent, run_input, close_on_cancel="fail", expose_errors=False, on_oversize=None, before_close=None)` in `iq_stream_langchain.agui` is the AG-UI relay.

### Which close line

| The run | Last frame | Close line |
| --- | --- | --- |
| finishes | the run's last part | `end`, from the `with` block, after your save |
| pauses on `interrupt()` | `updates` with `__interrupt__` | `end`: the answer waits for a human and resumes as a new run (`Command(resume=...)`) on a new stream |
| raises | `error` part with the exception's type | `fail`, from the `with` block; the exception goes on to your code |
| is cancelled: a cancelled task, `KeyboardInterrupt`, a drain request (`RunControl.request_drain()`) | `error` part named `Cancelled`, at once | `fail`, sent by the relay at once; with `close_on_cancel="end"`, `end`, sent by the relay once the graph has stopped |
| is left early: a `break`, `close()` / `aclose()` of the relay | `error` part named `Cancelled`, at once | `fail`, sent by the relay; with `close_on_cancel="end"`, `end` from the `with` block |

Except for a cancelled run with `close_on_cancel="end"`, the adapter never closes a stream with `end` itself: the `with` block does, so that what a late browser loads from you is saved before any browser is told the answer is complete.

"At once" means before LangGraph has stopped the run: `arelay()` reads the graph's stream in a task of its own, so a cancel lands in the relay, which publishes it, and not inside LangGraph, which lets a cancel through only once its running nodes have stopped (a tool may finish its work first).

A generator cannot tell a `break` from an exception raised by your own code inside the loop: the page reads the `Cancelled` part for both. The close line is still `fail` after an exception, from the relay or, with `close_on_cancel="end"`, from your block. Catch your own errors inside the loop, or use `publish_run()`, when the page must tell them apart. A cancel of the task that lands in your loop body (while it awaits a send or a save) is told apart on Python 3.11 and later, and closes with `close_on_cancel`; on 3.10 it is handled as a `break`.

### Errors and limits

Publishing never breaks the run. A publisher that gives up (the server is unreachable, it closed the stream) goes inert, the relay stops encoding parts and goes on reading the run, so that the run and its persistence complete. With the core's `on_error="raise"`, the publishing error leaves the relay like any exception of the run. A part that cannot be encoded is dropped and logged once. Dates and times are ISO 8601 strings, a `timedelta` its seconds, UUIDs, decimals and paths strings, an `Enum` its value, NaN and infinities `null` (a browser's `JSON.parse` refuses the bare constants); any other value JSON cannot hold is encoded by its `repr`, never lost.

A part over the publisher's `max_frame_bytes` (64 KiB unless the app has a limit of its own): a `messages` part with text content, a large tool result usually, is sent with its content cut to fit, ending in `…[truncated]` and with `"truncated": true` in its metadata. An `updates` part that carries an interrupt is the only one that tells the page the run waits for a human, so it is never dropped: it is sent with each interrupt's `id`, its `value` with every long string cut to fit (keys, names and ids whole, so that `HumanInTheLoopMiddleware`'s `action_requests` still name the tool to approve), and `"truncated": true`; a value whose structure alone does not fit is `null`. The whole value stays in the thread's state. Any other part (`updates`, `values`, a `tool-finished` event) is not sent: the result reaches the page in the `messages` part, and the whole of it is in the state your code saves.

On the AG-UI path, a text or arguments delta is sent as several deltas of the same id, a tool result is cut, and a snapshot is not sent. The interrupt (`ag_ui_langgraph`'s `on_interrupt` event, or the interrupt outcome of `RUN_FINISHED`) is cut as above; `RUN_FINISHED` and `RUN_ERROR` are never dropped: `RUN_FINISHED` goes without its outcome if even the interrupts' ids do not fit, and an error message exposed with `expose_errors=True` that does not fit goes masked, as on the native path.

## Limits worth knowing

- A stream is one run. A frame is at most 65,536 bytes and a stream at most 10 MiB and one hour, unless the app has limits of its own on the server. The server counts each frame at more than its size (its text as subscribers receive it, rounded up, plus 32 bytes), so a 430-byte token frame costs about 540 bytes: some 19,000 of them fit. When the server closes a stream, the relay stops publishing and the run goes on.
- On Python 3.10, an async run of an agent whose nodes call the model without passing their config, as `create_agent`'s do, streams whole messages instead of tokens: LangChain can hand the run's callbacks to such a call only from Python 3.11, whose `asyncio.create_task` takes a context. Synchronous runs stream tokens on every version; so does async code that passes `config` to the model. For the same reason an async run of `HumanInTheLoopMiddleware` (or any `interrupt()` called without the config at hand) fails on 3.10 in LangGraph itself, with "Called get_config outside of a runnable context".
- `useStream` ignores the `tools` parts (its tool events are the JavaScript runtime's), and so does `@ai-sdk/langchain`: tool calls and results still render from `messages`. Server features of `useStream` (history, branching, `joinStream`) need LangGraph Server.
- Graphs served by LangGraph Server or LangGraph Platform are not covered: those stream themselves. An application that only calls `invoke()` has nothing to publish: switch to `stream()` or `publish_run()`.
- `ag-ui-langgraph` 0.0.47 drops the arguments carried by the chunk that opens a tool call; providers that send a tool call's name and id first, with empty arguments, are not affected. Its `TOOL_CALL_RESULT` does not say that a tool failed, so a failed tool's error text goes out in it as the tool wrote it; the snapshots mask it.
- `iq_stream_langchain.experimental.iq_transformer(publisher, methods=...)` publishes the events of `stream_events(version="v3")`, which LangGraph marks as beta, while your code reads `run.output`. No browser client reads those events without LangGraph Server. A run your code stops early (a loop over a projection left, `run.abort()`, a cancelled task) looks to a transformer like one that finished, or is not reported to it at all: send the `Cancelled` part and the close line yourself.

```python
from iq_stream import IQClient
from iq_stream_langchain import CANCELLED_PART
from iq_stream_langchain.experimental import iq_transformer

iq = IQClient()
ticket = iq.ticket()
with iq.publisher(ticket.stream_id) as publisher:
    with agent.stream_events(question, config, version="v3", transformers=[iq_transformer(publisher)]) as run:
        for count, message in enumerate(run.messages):
            if count == 1:  # your own reason to stop
                publisher.send(CANCELLED_PART)
                publisher.close("fail")
                break
```

## Requirements

- Python 3.10 to 3.13, `iq-stream` 0.1, `langgraph` 1.1 or a later 1.x (the `tools` parts and drain requests need 1.2, the experimental module 1.2). For the `ag-ui` extra, `ag-ui-langgraph` 0.0.47 or later with FastAPI.
- Tested with `langgraph` 1.2.21, `langgraph-prebuilt` 1.1.1, `langchain` 1.5.0, `langchain-core` 1.8.0 and `ag-ui-langgraph` 0.0.47 (on Python 3.10: `langchain` 1.4.10 and `langchain-core` 1.7.7, the last releases that support it), and at the floor, on Python 3.10, with `langgraph` 1.1.0, `langgraph-prebuilt` 1.0.8 (the only release of the range `langgraph` 1.1.0 asks for that imports with it), `langchain` 1.2.11 (the first that takes `langgraph` 1.1) and `ag-ui-langgraph` 0.0.47.
- An IQ-Stream server 1.x.

## License

MIT
