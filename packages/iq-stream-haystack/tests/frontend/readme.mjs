// What a page around one of the README's browser snippets would do, in Node: the snippet's fetch() calls go to the
// application's origin (APP_URL), and the stream is read with Node's EventSource (--experimental-eventsource).
// tests/integration/test_live_hayhooks.py writes the snippet into a module that imports this one first, and runs it:
//
//   APP_URL=http://127.0.0.1:<port> node --experimental-eventsource <module>
//
// The module prints, as JSON, what the page posted and what StreamAgent or useChat's AbstractChat rebuilt.
import { AbstractChat } from 'ai';

const posted = [];
const fromThePage = globalThis.fetch;
globalThis.fetch = async (url, init) => {
  posted.push({ url: String(url), body: JSON.parse(init?.body ?? 'null') });
  return fromThePage(new URL(url, process.env.APP_URL), init);
};

const seen = { started: null, finished: null, error: null };

/** Records how the run started and ended, as AG-UI reports them to the agent's subscribers. */
export function tap(agent) {
  agent.subscribe({
    onRunStartedEvent: ({ event }) => { seen.started = { threadId: event.threadId, runId: event.runId }; },
    onRunFinishedEvent: ({ event }) => { seen.finished = { outcome: event.outcome ?? null }; },
    onRunErrorEvent: ({ event }) => { seen.error = event.message; },
  });
}

export function report(agent) {
  const assistants = agent.messages.filter((m) => m.role === 'assistant');
  process.stdout.write(JSON.stringify({
    posted,
    ...seen,
    texts: assistants.map((m) => m.content ?? '').filter(Boolean),
    toolCalls: assistants.flatMap((m) => m.toolCalls ?? [])
      .map((call) => ({ id: call.id, name: call.function.name, args: JSON.parse(call.function.arguments) })),
    toolResults: agent.messages.filter((m) => m.role === 'tool').map((m) => ({ id: m.toolCallId, content: m.content })),
  }));
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

export async function chat(transport) {
  const conversation = new HeadlessChat({ id: 'chat-1', transport });
  await conversation.sendMessage({ text: 'Weather in Paris?' });
  const parts = conversation.messages.at(-1)?.parts ?? [];
  process.stdout.write(JSON.stringify({
    posted,
    status: conversation.status,
    error: conversation.error ? String(conversation.error.message ?? conversation.error) : null,
    texts: parts.filter((p) => p.type === 'text').map((p) => p.text),
    tools: parts.filter((p) => p.type === 'dynamic-tool' || p.type.startsWith('tool-'))
      .map((p) => ({ id: p.toolCallId, state: p.state, input: p.input ?? null, output: p.output ?? null })),
  }));
}
