# iq-stream-python

Python packages for the [IQ-Stream server](https://github.com/neuron-core/iq-stream), which relays an agent's answer from a backend to browsers over Server-Sent Events. The browser side, and the TypeScript publisher, are in [iq-stream-js](https://github.com/neuron-core/iq-stream-js).

| Package | Import | |
| --- | --- | --- |
| [`iq-stream`](packages/iq-stream/README.md) | `iq_stream` | The core client: tokens, tickets and the publisher, in synchronous code and in asyncio, and the testing kit the adapters use |
| `iq-stream-langchain` | `iq_stream_langchain` | The adapter for LangChain and LangGraph. Coming |
| `iq-stream-agno` | `iq_stream_agno` | The adapter for Agno. Coming |
| `iq-stream-crewai` | `iq_stream_crewai` | The adapter for CrewAI. Coming |
| `iq-stream-llamaindex` | `iq_stream_llamaindex` | The adapter for LlamaIndex. Coming |
| `iq-stream-haystack` | `haystack_integrations.iq_stream` | The adapter for Haystack. Coming |

## How the packages relate

Each adapter is a thin layer over its framework's stream or callbacks and the core's publisher: it turns the framework's events into frames and calls `send()` and `close()`. The core owns everything that touches the wire (tokens, the held request, keepalives, resuming, the close line, and what to do with each answer of the server), so a fix to it reaches every adapter. An adapter depends on `iq-stream>=0.1.0,<0.2` and on its framework, and on nothing else; optional integrations such as AG-UI are extras.

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

CI (`.github/workflows/ci.yml`) runs, for every `packages/*/pyproject.toml`, lint, types, unit tests and the build on each Python its classifiers name, and the integration tests on the lowest and the highest; `scripts/ci_matrix.py` reads the list. CI pulls the server image with the workflow's own token, so the package on GitHub must give this repository read access (its settings, "Manage Actions access"), or be public.

### Adding an adapter

1. `packages/<name>/` with `pyproject.toml` (hatchling, the classifiers of the Pythons it supports, `iq-stream>=0.1.0,<0.2`, the `tool.uv.sources` entry above), `README.md`, a copy of `LICENSE`, `src/<import path>/py.typed`, `tests/conftest.py` with the core's pytest plugin, `tests/unit/` and `tests/integration/` (test files with names of their own: the two directories share one import path), then `uv lock`.
2. Its entry in `.github/dependabot.yml`, and its row in the table above.

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
