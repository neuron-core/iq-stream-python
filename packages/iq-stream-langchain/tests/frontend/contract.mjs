// Feeds what a real IQ subscriber received to the real front-end libraries and prints what they rebuilt, as JSON.
//
//   node contract.mjs <usestream|usechat|agui> <recording.json>
//   node contract.mjs transport <recording.json> <README.md>
//
// The recording is the subscriber's frames and terminal event. A FakeEventSource replays them into the real
// frames() of @neuron-core/iq-stream, so everything from the IQ client up is the code a page runs: the bridges of
// langgraph.mjs (the README's), StreamManager (the engine of useStream), chatTransport with AbstractChat (the class
// behind useChat), and StreamAgent through runAgent() of @ag-ui/client.
import { readFileSync } from 'node:fs';
import { StreamAgent } from '@neuron-core/iq-stream/ag-ui';
import { chatTransport } from '@neuron-core/iq-stream/ai-sdk';
import { FakeEventSource } from '@neuron-core/iq-stream/client/testing';
import { MessageTupleManager, StreamManager } from '@langchain/langgraph-sdk/ui';
import { AbstractChat } from 'ai';
import { langGraphParts, uiMessageChunks, useStreamEvents } from './langgraph.mjs';

const [check, file, readme] = process.argv.slice(2);
const recording = JSON.parse(readFileSync(file, 'utf8'));
const URL = 'http://iq.test/stream?token=t';

/** An EventSource that plays the recording back: open, every frame with its id, the terminal event. */
class Replay extends FakeEventSource {
  constructor(url) {
    super(url);
    setTimeout(() => {
      this.open();
      for (const frame of recording.frames) {
        this.frame(frame.data, { id: frame.id, event: frame.event === 'message' ? undefined : frame.event });
      }
      if (recording.terminal) this.terminal(recording.terminal.event, recording.terminal.data);
    }, 0);
  }
}

/** What StreamManager rebuilds from the events `open(signal)` returns, started as useStream starts a submit. */
async function rebuildStream(open) {
  const manager = new StreamManager(new MessageTupleManager(), { throttle: false });
  let finished = false;
  let error = null;
  await manager.start(open, {
    getMessages: (values) => (Array.isArray(values.messages) ? values.messages : []),
    setMessages: (current, messages) => ({ ...current, messages }),
    initialValues: {},
    callbacks: {},
    onSuccess: () => {
      finished = true;
      return undefined;
    },
    onError: (failure) => {
      error = { name: failure?.name ?? null, message: failure?.message ?? String(failure) };
    },
  });
  const values = manager.values ?? {};
  return {
    finished,
    error,
    interrupt: values.__interrupt__ ?? null,
    messages: (values.messages ?? []).map((message) => ({
      type: message.type,
      content: message.content,
      tool_calls: (message.tool_calls ?? []).map((call) => ({ name: call.name, args: call.args, id: call.id })),
    })),
  };
}

const useStream = () => rebuildStream(async () => useStreamEvents(langGraphParts(URL, { EventSource: Replay })));

/**
 * The README's useStream transport, called as useStream 1.16 calls it (react/stream.custom.js): for
 * thread.submit({ messages }), then for thread.submit(undefined, { command: { resume } }), whose events are read
 * from the recording. Prints what it posted to the backend and what StreamManager rebuilt.
 */
async function useStreamTransport() {
  const block = readFileSync(readme, 'utf8').match(/```ts\n(import \{ useStream \}[\s\S]*?)```/)[1];
  const code = block
    .split('\n')
    .filter((line) => !line.startsWith('import ') && !line.startsWith('const thread = '))
    .join('\n');
  const posted = [];
  const fetch = async (url, init) => {
    posted.push({ url, body: JSON.parse(init.body) });
    return { json: async () => ({ url: URL }) };
  };
  const transport = new Function('fetch', 'useStreamEvents', 'langGraphParts', `${code}\nreturn transport;`)(
    fetch,
    useStreamEvents,
    langGraphParts,
  );
  globalThis.EventSource = Replay;
  const config = { configurable: { thread_id: 'thread-1' } };
  const question = { messages: [{ type: 'human', content: 'Book a trip to Turin' }] };
  await transport.stream({ input: question, config, signal: new AbortController().signal });
  const resumed = await rebuildStream((signal) =>
    transport.stream({ input: undefined, command: { resume: 'yes' }, config, signal }),
  );
  return { posted, ...resumed };
}

class HeadlessChat extends AbstractChat {
  constructor(init) {
    super({
      ...init,
      state: {
        status: 'ready',
        error: undefined,
        messages: [],
        pushMessage(message) {
          this.messages = [...this.messages, message];
        },
        popMessage() {
          this.messages = this.messages.slice(0, -1);
        },
        replaceMessage(index, message) {
          this.messages = [...this.messages.slice(0, index), message, ...this.messages.slice(index + 1)];
        },
        snapshot: (value) => structuredClone(value),
      },
    });
  }
}

async function useChat() {
  const chunks = [];
  const transport = chatTransport({
    start: async () => ({ url: URL }),
    open: (ticket, signal) =>
      uiMessageChunks(ticket.url, { signal, EventSource: Replay }).pipeThrough(
        new TransformStream({
          transform(chunk, controller) {
            chunks.push(chunk.type);
            controller.enqueue(chunk);
          },
        }),
      ),
  });
  const chat = new HeadlessChat({ transport });
  await chat.sendMessage({ text: 'weather in Turin?' });
  const answer = chat.messages.at(-1);
  return {
    status: chat.status,
    error: chat.error?.message ?? null,
    chunks,
    role: answer?.role,
    parts: (answer?.parts ?? [])
      .filter((part) => part.type === 'text' || part.type === 'dynamic-tool')
      .map((part) =>
        part.type === 'text'
          ? { type: 'text', text: part.text }
          : { type: part.type, toolName: part.toolName, state: part.state, input: part.input, output: part.output },
      ),
  };
}

async function agUi() {
  globalThis.EventSource = Replay; // StreamAgent's default open reads IQ frames through EventSource
  const agent = new StreamAgent({ threadId: 'thread-1', start: async () => ({ url: URL }) });
  agent.messages = [{ id: 'u1', role: 'user', content: 'weather in Turin?' }];
  const events = [];
  const failures = [];
  let runError = null;
  let outcome = null;
  try {
    await agent.runAgent(
      {},
      {
        onEvent: ({ event }) => {
          events.push(event.type);
        },
        onRunErrorEvent: ({ event }) => {
          runError = event.message;
        },
        onRunFinishedEvent: ({ event }) => {
          outcome = event.outcome ?? null;
        },
        onRunFailed: ({ error }) => {
          failures.push(error.message);
        },
      },
    );
  } catch (error) {
    failures.push(error.message);
  }
  return {
    events,
    failures,
    runError,
    outcome,
    messages: agent.messages.map((message) => ({
      role: message.role,
      content: message.content ?? null,
      toolCalls: (message.toolCalls ?? []).map((call) => ({ name: call.function.name, arguments: call.function.arguments })),
      toolCallId: message.toolCallId ?? null,
    })),
  };
}

const checks = { usestream: useStream, transport: useStreamTransport, usechat: useChat, agui: agUi };
process.stdout.write(`${JSON.stringify(await checks[check]())}\n`);
