# iq-stream-python

Python packages for the [IQ-Stream server](https://github.com/neuron-core/iq-stream), which relays an agent's answer from a backend to browsers over Server-Sent Events. The browser side, and the TypeScript publisher, are in [iq-stream-js](https://github.com/neuron-core/iq-stream-js).

| Package | Import | Python | |
| --- | --- | --- | --- |
| [`iq-stream`](packages/iq-stream/README.md) | `iq_stream` | 3.10 to 3.13 | The core client: tokens, tickets and the publisher, in synchronous code and in asyncio, and the testing kit the adapters use |
| [`iq-stream-langchain`](packages/iq-stream-langchain/README.md) | `iq_stream_langchain` | 3.10 to 3.13 | LangGraph graphs and LangChain `create_agent` agents, as LangGraph stream parts (read by `useStream`, and by the AI SDK's `useChat` through `@ai-sdk/langchain`), or as AG-UI events with the `[ag-ui]` extra |
| [`iq-stream-agno`](packages/iq-stream-agno/README.md) | `iq_stream_agno` | 3.10 to 3.13 | Agno agents, teams and workflows, as Agno's own event JSON or as AG-UI events (`[ag-ui]`), and an AgentOS interface (`[os]`) |
| [`iq-stream-crewai`](packages/iq-stream-crewai/README.md) | `iq_stream_crewai` | 3.10 to 3.13 | CrewAI crews, flows and agents, as AG-UI events or as CrewAI's own events |
| [`iq-stream-llamaindex`](packages/iq-stream-llamaindex/README.md) | `iq_stream_llamaindex` | 3.10 to 3.13 | LlamaIndex workflows, and agents with `[agents]`, as LlamaIndex's own events or as AG-UI events (`[ag-ui]`) |
| [`iq-stream-haystack`](packages/iq-stream-haystack/README.md) | `haystack_integrations.iq_stream` | 3.10 to 3.13 | Haystack agents and pipelines, as AG-UI events, AI SDK UI message chunks or Haystack's own chunks, and helpers for Hayhooks (`[hayhooks]`) |

The browser reads every format above with [`@neuron-core/iq-stream`](https://www.npmjs.com/package/@neuron-core/iq-stream): `StreamAgent` for AG-UI clients such as CopilotKit, `chatTransport` for the AI SDK's `useChat`, and the plain subscriber for the frameworks' own formats.

## How the packages relate

Each adapter is a thin layer over its framework's stream or callbacks and the core's publisher: it turns the framework's events into frames and calls `send()` and `close()`. The core owns everything that touches the wire (tokens, the held request, keepalives, resuming, the close line, and what to do with each answer of the server), so a fix to it reaches every adapter. An adapter depends on `iq-stream>=0.1.0,<0.2` and on its framework, and on nothing else; optional integrations such as AG-UI are extras.

The adapters follow the same rules and share their vocabulary, so moving from one framework to another changes the wrapping, not the behaviour. The stream closes with `end` only once the run completed and the answer was saved (inside the adapter's block or loop, or in its `before_close` hook, both of which run before the close line), and with `fail` otherwise. A cancelled run closes with `fail`, or with `end` when `close_on_cancel="end"` says the application keeps what was published. Where the framework stops a run to wait for a human, the leg ends with `end` and the browser is told why. Error text reaches the browser masked as "An error occurred." unless `expose_errors=True`. Frames over the server's limit are cut, split or left out as each README describes, never sent to be refused. Publishing never breaks the run: a frame that cannot be sent is dropped and reported to the adapter's hook where it has one (`on_dropped` in Agno and Haystack, `on_oversize` in LangChain), else logged without frame data, and a publisher that gives up leaves the run to be read to its end. `on_error` keeps the core's meaning everywhere: with a client opened with `on_error="raise"`, the publisher's error reaches the application, from the loop where the adapter wraps the framework's iterator (LangChain, Agno, LlamaIndex) and when the block is left where it wraps the run in a block (CrewAI, Haystack).

Each package is a project of its own, with its own `pyproject.toml`, `uv.lock` and virtual environment, and there is deliberately no uv workspace: a workspace resolves one set of versions for every member, and the frameworks pin conflicting versions of the same libraries (CrewAI resolves `pydantic 2.12.5` and `openai 2.x`, Haystack `openai 3.x`). One lockfile per package also lets each be tested against exactly the framework versions it was built for. During development an adapter takes the core from this repository:

```toml
[tool.uv.sources]
iq-stream = { path = "../iq-stream", editable = true }
```

Build backends ignore `tool.uv.sources`, so the wheel an adapter publishes depends on `iq-stream` from PyPI.

## Development

You need [uv](https://docs.astral.sh/uv/), Python 3.10 to 3.13, and Docker for the integration tests. From a package's directory:

```sh
uv sync                                    # .venv with the dev group, from uv.lock
uv run ruff check . && uv run ruff format --check .
uv run pyright src                         # strict
uv run pytest tests/unit                   # no network, no Docker
uv run pytest tests/integration            # against the two servers below
uv build                                   # sdist and wheel into dist/
uv run --python 3.10 pytest tests/unit     # on another Python (uv rebuilds .venv for it)
```

The integration tests run against two real IQ-Stream servers: one with default settings on port 8080 and one with short timeouts and small limits on port 8081 (the settings of `iq_stream.testing.SHORT_SETTINGS`). Start and stop them from the repository's root:

```sh
python3 scripts/servers.py start           # in Docker, from IQ_STREAM_TEST_IMAGE (default ghcr.io/neuron-core/iq-stream:1)
python3 scripts/servers.py stop
```

The tests find servers by themselves: the ones `IQ_STREAM_TEST_URL` and `IQ_STREAM_TEST_SHORT_URL` name, else those answering on the two ports, else containers they start and remove. `IQ_STREAM_TEST_APP_KEY` and `IQ_STREAM_TEST_APP_SECRET` override the credentials (`demo-app`, `test-app-secret-0123456789abcdef0123`). The tests that kill or restart a server start a container of their own on a free port. An adapter's suite gets the same servers through the core's pytest plugin (`pytest_plugins = ["iq_stream.testing.pytest_plugin"]` in its `tests/conftest.py`); see the core's README.

CI (`.github/workflows/ci.yml`) runs, for every `packages/*/pyproject.toml`, lint, types, unit tests and the build on each Python its classifiers name, and the integration tests on the lowest and the highest; `scripts/ci_matrix.py` reads the list. It also runs each package's unit tests on its lowest Python against the lowest versions of its dependencies, which the package declares in `[tool.iq-stream] floor` (`uv sync --locked`, then `uv pip install` of those pins, then `uv run --no-sync pytest tests/unit`), so that the ranges published on PyPI are ranges that ran. The adapters' integration jobs check out [iq-stream-js](https://github.com/neuron-core/iq-stream-js) and build its browser package, so that their front-end contract tests feed what a real subscriber received to the real browser library. Locally those tests skip unless `IQ_STREAM_JS_DIR` points at such a checkout (`npm ci && npm run build -w packages/iq-stream` in it) and `node` is on the `PATH`. CI pulls the server image with the workflow's own token, so the package on GitHub must give this repository read access (its settings, "Manage Actions access"), or be public.

### Adding an adapter

1. `packages/<name>/` with `pyproject.toml` (hatchling, the classifiers of the Pythons it supports, `iq-stream>=0.1.0,<0.2`, the `tool.uv.sources` entry above), `README.md`, a copy of `LICENSE`, `src/<import path>/py.typed`, `tests/conftest.py` with the core's pytest plugin, `tests/unit/` and `tests/integration/` (test files with names of their own: the two directories share one import path), then `uv lock`.
2. In its `pyproject.toml`, the lowest version of each dependency whose range it states, which CI runs its unit tests against:

   ```toml
   [tool.iq-stream]
   floor = ["agno==2.9.0", "ag-ui-protocol==0.1.15"]
   ```

   Pin the versions you develop against in `uv.lock` only (`uv lock --upgrade-package`), not in `[tool.uv] constraint-dependencies`: `uv pip install` applies those from the package's directory, and an exact pin there makes the floor unsatisfiable.
3. Its entry in `.github/dependabot.yml`, and its row in the table above.

CI and releases pick it up from its directory.

## Releasing

Each package is released by itself, from GitHub Actions (`.github/workflows/release.yml`), when a tag `<package>@<version>` is pushed:

```sh
# set version = "0.1.1" in packages/iq-stream/pyproject.toml, then relock every package:
for package in packages/*/; do (cd "$package" && uv lock); done
git commit -am "iq-stream 0.1.1"
git tag iq-stream@0.1.1
git push origin main iq-stream@0.1.1
```

Every package, not only the one released: an adapter's `uv.lock` records the version of the core it takes from `../iq-stream`, and CI, which the release runs first, syncs each package with `--locked`.

The workflow checks the tag against the package's `pyproject.toml`, runs the whole of CI, builds with `uv build`, and publishes to PyPI by trusted publishing, with attestations. A pre-release version in PEP 440's terms (`0.2.0b1`, `0.2.0rc1`) is published as such: pip installs it only when asked for it. When an adapter needs a newer core, release the core first and raise the adapter's range.

Trusted publishing needs, once per package, a publisher on PyPI. For a package not yet on PyPI, add a pending publisher (your account, Publishing): project name (`iq-stream`, `iq-stream-agno`, ...), owner `neuron-core`, repository `iq-stream-python`, workflow `release.yml`, environment `pypi`. The repository needs the environment `pypi` (settings, Environments); add required reviewers to it to approve each release.

## License

MIT
