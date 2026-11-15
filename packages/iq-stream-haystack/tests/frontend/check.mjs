// Feeds what a real subscriber received to the browser side of @neuron-core/iq-stream and prints, as JSON, what the
// front-end libraries rebuild from it. Run by tests/integration/test_live_frontend.py from a directory whose
// node_modules is the JavaScript checkout's:
//
//   node check.mjs <recording.json>
//
// The recording is {protocol, frames: [{id, event, data}], terminal: {event, data}}. Its frames are replayed into the
// client's subscription through FakeEventSource, with the ids and event names the server sent:
//   "ag-ui":    StreamAgent (./ag-ui) driven by runAgent() of @ag-ui/client, which verifies the events and applies
//               them to the agent's messages, as CopilotKit drives an agent; every event also parsed by @ag-ui/core;
//   "ai-sdk":   chatTransport (./ai-sdk) under AbstractChat, the state machine of useChat; every chunk also validated
//               by the AI SDK's uiMessageChunkSchema;
//   "haystack": frames() (./client), the transport client alone, as a page without a chat framework reads it.
import { readFileSync } from 'node:fs';
import { EventSchemas } from '@ag-ui/core/schemas';
import { safeValidateTypes } from '@ai-sdk/provider-utils';
import { StreamAgent } from '@neuron-core/iq-stream/ag-ui';
import { chatTransport } from '@neuron-core/iq-stream/ai-sdk';
import { frames } from '@neuron-core/iq-stream/client';
import { FakeEventSource } from '@neuron-core/iq-stream/client/testing';
import { AbstractChat, uiMessageChunkSchema } from 'ai';

const recording = JSON.parse(readFileSync(process.argv[2], 'utf8'));

/** A source that replays the recording once the client listens to it. */
function replay() {
  const source = new FakeEventSource('replay');
  setImmediate(() => {
    source.open();
    for (const frame of recording.frames) {
      source.frame(frame.data, { id: frame.id, event: frame.event === 'message' ? undefined : frame.event });
    }
    source.terminal(recording.terminal.event, recording.terminal.data);
  });
  return source;
}

const message = (error) => String(error?.message ?? error);

/** A tool call's arguments as an object, or as the text they are when a broken stream cut them off. */
function args(text) {
  try {
    return JSON.parse(text || '{}');
  } catch {
    return text;
  }
}

async function agUi() {
  const invalid = [];
  for (const frame of recording.frames) {
    const parsed = EventSchemas.safeParse(JSON.parse(frame.data));
    if (!parsed.success) invalid.push(frame.data);
  }
  const agent = new StreamAgent({
    threadId: 'thread-1',
    start: async () => ({ url: 'replay' }),
    open: () => frames(() => replay(), { parse: JSON.parse }),
  });
  agent.messages = [{ id: 'user-1', role: 'user', content: 'Weather in Paris?' }];
  const types = [];
  const failures = [];
  let finished = null;
  let runError = null;
  try {
    await agent.runAgent({}, {
      onEvent: ({ event }) => { types.push(event.type); },
      onRunFinishedEvent: ({ event }) => { finished = { outcome: event.outcome ?? null }; },
      onRunErrorEvent: ({ event }) => { runError = { message: event.message, code: event.code ?? null }; },
      onRunFailed: ({ error }) => { failures.push(message(error)); },
    });
  } catch (error) {
    failures.push(message(error));
  }
  const assistants = agent.messages.filter((m) => m.role === 'assistant');
  return {
    invalid,
    types,
    roles: agent.messages.map((m) => m.role),
    texts: assistants.map((m) => m.content ?? '').filter(Boolean),
    reasoning: agent.messages.filter((m) => m.role === 'reasoning').map((m) => m.content),
    toolCalls: assistants.flatMap((m) => m.toolCalls ?? [])
      .map((call) => ({ id: call.id, name: call.function.name, args: args(call.function.arguments) })),
    toolResults: agent.messages.filter((m) => m.role === 'tool').map((m) => ({ toolCallId: m.toolCallId, content: m.content })),
    finished,
    runError,
    failures,
  };
}

/** AbstractChat with a plain state: the engine of useChat without React. */
class HeadlessChat extends AbstractChat {
  constructor(init) {
    super({
      ...init,
      state: {
        status: 'ready',
        error: undefined,
        messages: [],
        pushMessage(m) { this.messages = [...this.messages, m]; },
        popMessage() { this.messages = this.messages.slice(0, -1); },
        replaceMessage(index, m) { this.messages = [...this.messages.slice(0, index), m, ...this.messages.slice(index + 1)]; },
        snapshot: (value) => structuredClone(value),
      },
    });
  }
}

async function aiSdk() {
  const invalid = [];
  const types = [];
  const transport = chatTransport({
    start: async () => ({ url: 'replay' }),
    open: (ticket, signal) => frames(() => replay(), { parse: JSON.parse, signal }).pipeThrough(new TransformStream({
      async transform(chunk, controller) {
        types.push(chunk.type);
        const checked = await safeValidateTypes({ value: chunk, schema: uiMessageChunkSchema });
        if (!checked.success) invalid.push(JSON.stringify(chunk));
        controller.enqueue(chunk);
      },
    })),
  });
  let finish = null;
  const chat = new HeadlessChat({ transport, onFinish: ({ isAbort, isError }) => { finish = { isAbort, isError }; } });
  await chat.sendMessage({ text: 'Weather in Paris?' });
  const answer = chat.messages.at(-1);
  const parts = answer?.role === 'assistant' ? answer.parts : [];
  return {
    invalid,
    types,
    status: chat.status,
    error: chat.error ? message(chat.error) : null,
    finish,
    messageId: answer?.id ?? null,
    texts: parts.filter((p) => p.type === 'text').map((p) => p.text),
    reasoning: parts.filter((p) => p.type === 'reasoning').map((p) => p.text),
    steps: parts.filter((p) => p.type === 'step-start').length,
    tools: parts.filter((p) => p.type.startsWith('tool-') || p.type === 'dynamic-tool').map((p) => ({
      type: p.type,
      toolCallId: p.toolCallId,
      state: p.state,
      input: p.input ?? null,
      output: p.output ?? null,
      errorText: p.errorText ?? null,
    })),
    data: parts.filter((p) => p.type.startsWith('data-')).map((p) => ({ type: p.type, data: p.data })),
  };
}

async function haystack() {
  const events = [];
  let outcome = { kind: 'end' };
  const reader = frames(() => replay(), { parse: JSON.parse }).getReader();
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      events.push(value);
    }
  } catch (error) {
    outcome = { kind: error.kind ?? null, reason: error.reason ?? null };
  }
  const chunks = events.filter((e) => e.type === 'chunk');
  const calls = new Map();
  for (const chunk of chunks) {
    for (const delta of chunk.tool_calls ?? []) {
      const call = calls.get(`${delta.index}`) ?? { id: null, name: null, arguments: '' };
      call.id ??= delta.id;
      call.name ??= delta.tool_name;
      call.arguments += delta.arguments ?? '';
      calls.set(`${delta.index}`, call);
    }
  }
  return {
    types: events.map((e) => e.type),
    text: chunks.map((c) => c.content).join(''),
    toolCalls: [...calls.values()].map((c) => ({ id: c.id, name: c.name, args: JSON.parse(c.arguments || '{}') })),
    toolResults: chunks.filter((c) => c.tool_call_result)
      .map((c) => ({ id: c.tool_call_result.origin.id, result: c.tool_call_result.result, error: c.tool_call_result.error })),
    last: events.at(-1) ?? null,
    outcome,
  };
}

const checks = { 'ag-ui': agUi, 'ai-sdk': aiSdk, haystack };
process.stdout.write(JSON.stringify(await checks[recording.protocol]()));
