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

`arelay()` and `apublish_run()` are the same over `graph.astream()`. The route answers with the ticket at once; the run is published while the page reads it. A second route cancels it.

```python
import asyncio

from fastapi import FastAPI
from pydantic import BaseModel

from iq_stream import AsyncPublisher, IQClient
from iq_stream_langchain import apublish_run

app = FastAPI()
iq = IQClient()
running: dict[str, asyncio.Task[None]] = {}  # stream id -> the task that publishes it


class ChatIn(BaseModel):
    thread_id: str
    message: str


@app.post("/chat")
async def chat(body: ChatIn) -> dict[str, str | int]:
    ticket = iq.ticket()
    publisher = iq.apublisher(ticket.stream_id)  # the request to the IQ server opens now
    task = asyncio.create_task(run_agent(publisher, body))
    running[ticket.stream_id] = task
    task.add_done_callback(lambda _: running.pop(ticket.stream_id, None))
    return ticket.to_dict()  # streamId, token, url, expiresAt


@app.post("/chat/{stream_id}/stop")
async def stop(stream_id: str) -> None:
    if task := running.get(stream_id):
        task.cancel()  # the page gets the Cancelled part, then iq:fail


@app.get("/chat/{thread_id}")
async def history(thread_id: str) -> list[dict[str, str]]:
    """What a page loads after iq:gone: the thread as the checkpointer saved it."""
    state = await agent.aget_state({"configurable": {"thread_id": thread_id}})
    return [{"type": message.type, "content": message.text} for message in state.values.get("messages", [])]


async def run_agent(publisher: AsyncPublisher, body: ChatIn) -> None:
    config = {"configurable": {"thread_id": body.thread_id}}
    question = {"messages": [{"role": "user", "content": body.message}]}
    async with publisher:  # "end"; "fail" on an exception or a cancellation
        await apublish_run(publisher, agent, question, config)
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

By default the modes are `messages`, `updates`, `custom` and `tools` (`DEFAULT_STREAM_MODE`). `values` is left out because each part repeats the whole state, which grows with the thread. A `messages` part keeps the metadata keys the browser clients read (`SLIM_METADATA`), about 430 bytes a token instead of 720.

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
  stop: (ticket) => void fetch(`/chat/${ticket.streamId}/stop`, { method: 'POST' }),
  open: (ticket, signal) => uiMessageChunks(ticket.url, { signal }),
});

const { messages, sendMessage, stop, status } = useChat({ id: threadId, transport });
// A tool call is a part of type 'dynamic-tool', with toolName, input, output and state.
```

With LangGraph's `useStream` (`npm install @neuron-core/iq-stream @langchain/langgraph-sdk`):

```ts
import { useStream } from '@langchain/langgraph-sdk/react';

const thread = useStream({
  messagesKey: 'messages',
  transport: {
    async stream({ input, signal }) {
      const message = input.messages.at(-1).content;
      const response = await fetch('/chat', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ thread_id: threadId, message }), signal,
      });
      const ticket = await response.json();
      return useStreamEvents(langGraphParts(ticket.url, { signal }));
    },
  },
});
```

A failed run reaches `useStream` as a `StreamError` named after the exception type, and a cancelled one as one named `Cancelled`; `useChat` gets an `error` chunk or an `abort` chunk. When a subscription ends with `iq:gone` (the stream is over and frames this tab missed are gone), load the thread from your backend, as the `/chat/{thread_id}` route above offers. The functions above and the transports ran in Node against what a real subscriber received, through `StreamManager` (the engine of `useStream`), `AbstractChat` (the class behind `useChat`) and `@ai-sdk/langchain` 3.0.172; the React hooks themselves were not run.

## AG-UI

For a front end that speaks [AG-UI](https://docs.ag-ui.com) (CopilotKit, or `StreamAgent` of `@neuron-core/iq-stream/ag-ui`), relay the events of `ag_ui_langgraph.LangGraphAgent`. It owns the run: it needs a checkpointer and calls the graph itself.

```python
import asyncio

from ag_ui.core import RunAgentInput
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
    async with publisher:
        async for _ in arelay_agui(publisher, agui_agent, run_input):
            pass
```

Each event is one frame, its JSON as the AG-UI encoder writes it (camelCase). `ag_ui_langgraph` reports a failed run as a `RUN_ERROR` event instead of raising, so the relay closes the stream with `fail` as soon as it has sent one; its message is replaced by `An error occurred.` unless `expose_errors=True`. A cancelled run ends with `RUN_FINISHED` and the `cancelled` outcome, after the events that close what the client holds open (a message, a tool call, a step), which AG-UI clients require of a finished run. In the page, `new StreamAgent({ threadId, start: async (input) => (await fetch('/agent', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(input) })).json() })` reads these frames as they are.

Construct the agent with `emit_raw_events=False`: raw events multiply the size of a stream (69 events and 60 kB against 32 events and 4 kB for one run with a tool call).

## Reference

### `relay(publisher, graph, input, config=None, **options)` and `arelay(...)`

Run `graph.stream(input, config, subgraphs=True, version="v2")` (`astream` for `arelay`), publish each part whose mode is in `stream_mode` as one frame, and yield every part unchanged. `relay` takes a synchronous `Publisher`; `arelay` takes an `AsyncPublisher`, or a `Publisher` for graphs whose nodes block the loop and would starve the asynchronous publisher's keepalive. Both return generators: `close()` / `aclose()` cancel the run.

### `publish_run(publisher, graph, input, config=None, **options)` and `apublish_run(...)`

Relay the run to its end and return the final state of the root graph, its last `values`, which are requested from the graph and not published.

| Option | Default | |
| --- | --- | --- |
| `stream_mode` | `DEFAULT_STREAM_MODE` | The modes requested and published: `messages`, `updates`, `custom`, `tools`, and `values`, `tasks`, `checkpoints`, `debug` if you add them. |
| `also_yield` | `()` | Modes requested from the graph and yielded, not published. |
| `metadata` | `"slim"` | The metadata of a `messages` part: `"slim"` (the keys the browser clients read), `"full"`, or a list of keys. |
| `expose_errors` | `False` | Publish the message of a failed run's exception and of a `tool-error` part. Otherwise the browser reads `An error occurred.`. |
| `cancelled` | `"fail"` | The close line of a cancelled run, or `"end"` for an application that treats a stopped answer as final. |
| `on_oversize` | `None` | Called with an `Oversize` (`part_type`, `size`, `limit`, `truncated`) for every part over the frame limit. Without it, the first one of a run is logged as a warning on the `iq_stream_langchain` logger. |
| `**stream_kwargs` | | Passed to `graph.stream()`: `context`, `durability`, `interrupt_before`, `control`, ... |

`encode_part(part, metadata="slim", expose_errors=False)` and `error_part(error, expose=False)` return the frame data the relay would send, for code of your own. `arelay_agui(publisher, agent, run_input, cancelled="fail", expose_errors=False, on_oversize=None)` in `iq_stream_langchain.agui` is the AG-UI relay.

### Which close line

| The run | Last frame | Close line |
| --- | --- | --- |
| finishes | the run's last part | `end`, from the `with` block, after your save |
| pauses on `interrupt()` | `updates` with `__interrupt__` | `end`: the answer waits for a human and resumes as a new run (`Command(resume=...)`) on a new stream |
| raises | `error` part with the exception's type | `fail`, from the `with` block; the exception goes on to your code |
| is cancelled: a cancelled task, a loop left early, `KeyboardInterrupt`, a drain request (`RunControl.request_drain()`) | `error` part named `Cancelled` | `fail` (or `cancelled="end"`), sent by the relay itself |

The adapter never closes a stream that finished: the `with` block does, so that what a late browser loads from you is saved before any browser is told the answer is complete.

### Errors and limits

Publishing never breaks the run. A publisher that gives up (the server is unreachable, it closed the stream) goes inert, the relay stops encoding parts and goes on reading the run, so that the run and its persistence complete. With the core's `on_error="raise"`, the publishing error leaves the relay like any exception of the run. A part that cannot be encoded is dropped and logged once; a value JSON cannot hold is encoded by its `repr`, never lost.

A part over the publisher's `max_frame_bytes` (64 KiB unless the app has a limit of its own): a `messages` part with text content, a large tool result usually, is sent with its content cut to fit, ending in `…[truncated]` and with `"truncated": true` in its metadata. Any other part (`updates`, `values`, a `tool-finished` event) is not sent: the result reaches the page in the `messages` part, and the whole of it is in the state your code saves. On the AG-UI path, a text or arguments delta is sent as several deltas of the same id, a tool result is cut, and a snapshot is not sent.

## Limits worth knowing

- A stream is one run. A frame is at most 65,536 bytes and a stream at most 10 MiB and 30 minutes, unless the app has limits of its own on the server: about 24,000 token frames. When the server closes a stream, the relay stops publishing and the run goes on.
- On Python 3.10, an async run of an agent whose nodes call the model without passing their config, as `create_agent`'s do, streams whole messages instead of tokens: LangChain can hand the run's callbacks to such a call only from Python 3.11, whose `asyncio.create_task` takes a context. Synchronous runs stream tokens on every version; so does async code that passes `config` to the model.
- `useStream` ignores the `tools` parts (its tool events are the JavaScript runtime's), and so does `@ai-sdk/langchain`: tool calls and results still render from `messages`. Server features of `useStream` (history, branching, `joinStream`) need LangGraph Server.
- Graphs served by LangGraph Server or LangGraph Platform are not covered: those stream themselves. An application that only calls `invoke()` has nothing to publish: switch to `stream()` or `publish_run()`.
- `ag-ui-langgraph` 0.0.47 drops the arguments carried by the chunk that opens a tool call; providers that send a tool call's name and id first, with empty arguments, are not affected.
- `iq_stream_langchain.experimental.iq_transformer(publisher, methods=...)` publishes the events of `stream_events(version="v3")`, which LangGraph marks as beta, while your code reads `run.output`. No browser client reads those events without LangGraph Server.

## Requirements

- Python 3.10 to 3.13, `iq-stream` 0.1, `langgraph` 1.1 or later (the `tools` parts and drain requests need 1.2, the experimental module 1.2). For the `ag-ui` extra, `ag-ui-langgraph` 0.0.47 or later with FastAPI.
- Tested with `langgraph` 1.2.21, `langgraph-prebuilt` 1.1.1, `langchain` 1.5.0, `langchain-core` 1.8.0 and `ag-ui-langgraph` 0.0.47 (on Python 3.10: `langchain` 1.4.10 and `langchain-core` 1.7.7, the last releases that support it).
- An IQ-Stream server 1.x.

## License

MIT
