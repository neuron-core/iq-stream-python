"""The plan's server integration: the example Hayhooks wrapper under a real Hayhooks server, called over HTTP, its
stream read by a real subscriber, and by the README's browser snippets in Node (they need IQ_STREAM_JS_DIR)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from iq_stream.testing import TestServers, free_port, subscribe

from tests.checks import verify_agui
from tests.live import in_order_once, js_missing

PACKAGE = Path(__file__).resolve().parents[2]
EXAMPLES = PACKAGE / "examples" / "hayhooks"
SNIPPETS = re.findall(r"```ts\n(.*?)```", (PACKAGE / "README.md").read_text(encoding="utf-8"), re.DOTALL)


@pytest.fixture
def hayhooks(iq_servers: TestServers, tmp_path: Path) -> Iterator[tuple[str, Path]]:
    """A Hayhooks server on a free port, deploying the example wrapper; stopped after the test."""
    port = free_port()
    answers = tmp_path / "answers"
    env = {
        **os.environ,
        "IQ_STREAM_URL": iq_servers.default.url,
        "IQ_STREAM_PUBLIC_URL": iq_servers.default.url,
        "IQ_STREAM_APP_KEY": iq_servers.app_key,
        "IQ_STREAM_APP_SECRET": iq_servers.app_secret,
        "ANSWERS_DIR": str(answers),
        "HAYSTACK_TELEMETRY_ENABLED": "False",
    }
    command = [str(Path(sys.executable).parent / "hayhooks"), "run", "--host", "127.0.0.1", "--port", str(port),
               "--pipelines-dir", str(EXAMPLES), "--without-monitoring", "--without-tracing"]  # fmt: skip
    log = (tmp_path / "hayhooks.log").open("w")
    process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=tmp_path)
    url = f"http://127.0.0.1:{port}"
    try:
        deadline = time.monotonic() + 60
        while True:
            assert process.poll() is None, (tmp_path / "hayhooks.log").read_text()
            try:
                if httpx.get(f"{url}/status", timeout=1).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            assert time.monotonic() < deadline, (tmp_path / "hayhooks.log").read_text()
            time.sleep(0.2)
        yield url, answers
    finally:
        process.terminate()
        try:
            process.wait(10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(10)
        log.close()


def test_the_run_endpoint_answers_with_the_ticket_and_the_run_reaches_the_browser(
    hayhooks: tuple[str, Path], iq_servers: TestServers
) -> None:
    url, answers = hayhooks
    started = time.monotonic()
    response = httpx.post(f"{url}/weather_agent/run", json={"question": "Weather in Paris?"}, timeout=30)
    answered_in = time.monotonic() - started
    assert response.status_code == 200, response.text
    ticket = response.json()["result"]
    assert set(ticket) == {"streamId", "token", "url", "expiresAt"}
    assert answered_in < 1.0  # the tool alone takes a second: the endpoint did not wait for the run

    received = subscribe(ticket["url"], timeout=30)
    conversation = verify_agui(in_order_once(received))
    assert conversation.texts == ["Let me check the weather.", "It is sunny and 21C in Paris."]
    assert conversation.results == {"call_paris": "Sunny, 21C in Paris"}
    assert received.terminal is not None and received.terminal.event == "iq:end"
    # The answer was saved before the close line, so it is there when the browser hears iq:end.
    saved = json.loads((answers / f"{ticket['streamId']}.json").read_text())
    assert saved[-1]["content"][0]["text"] == "It is sunny and 21C in Paris."


def run_snippet(tmp_path: Path, app_url: str, module: str) -> dict[str, Any]:
    """Runs a README snippet as a page would, beside tests/frontend/readme.mjs, with the JavaScript checkout's
    node_modules; returns what it printed."""
    shutil.copy(PACKAGE / "tests" / "frontend" / "readme.mjs", tmp_path / "readme.mjs")
    (tmp_path / "node_modules").symlink_to(Path(os.environ["IQ_STREAM_JS_DIR"]) / "node_modules")
    (tmp_path / "page.mjs").write_text(module, encoding="utf-8")
    process = subprocess.run(
        ["node", "--experimental-eventsource", "--no-warnings", "page.mjs"],
        cwd=tmp_path,
        env={**os.environ, "APP_URL": app_url},
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return json.loads(process.stdout)


def test_the_readme_has_its_browser_snippets() -> None:
    assert len(SNIPPETS) == 2
    assert "new StreamAgent(" in SNIPPETS[0] and "await agent.runAgent();" in SNIPPETS[0]
    assert "chatTransport(" in SNIPPETS[1]


@pytest.mark.skipif(js_missing() is not None, reason=js_missing() or "")
def test_the_readme_stream_agent_runs_the_example_with_its_own_thread_and_run_ids(
    hayhooks: tuple[str, Path], tmp_path: Path
) -> None:
    url, answers = hayhooks
    snippet = SNIPPETS[0].replace("await agent.runAgent();", "tap(agent);\nawait agent.runAgent();")
    page = run_snippet(tmp_path, url, "import { report, tap } from './readme.mjs';\n" + snippet + "report(agent);\n")
    [posted] = page["posted"]
    assert posted["url"] == "/weather_agent/run"
    assert posted["body"]["question"] == "Weather in Paris?" and posted["body"]["thread_id"] == "chat-1"
    assert page["started"] == {"threadId": "chat-1", "runId": posted["body"]["run_id"]}  # the client's own ids
    assert page["finished"] == {"outcome": None} and page["error"] is None
    assert page["texts"] == ["Let me check the weather.", "It is sunny and 21C in Paris."]
    assert page["toolCalls"] == [{"id": "call_paris", "name": "get_weather", "args": {"city": "Paris"}}]
    assert page["toolResults"] == [{"id": "call_paris", "content": "Sunny, 21C in Paris"}]
    assert len(list(answers.glob("*.json"))) == 1  # saved before the close line


@pytest.mark.skipif(js_missing() is not None, reason=js_missing() or "")
def test_the_readme_chat_transport_runs_the_example_under_use_chat_s_engine(
    hayhooks: tuple[str, Path], tmp_path: Path
) -> None:
    url, _ = hayhooks
    page = run_snippet(
        tmp_path, url, "import { chat } from './readme.mjs';\n" + SNIPPETS[1] + "await chat(transport);\n"
    )
    [posted] = page["posted"]
    assert posted["body"] == {"question": "Weather in Paris?", "thread_id": "chat-1", "protocol": "ai-sdk"}
    assert page["status"] == "ready" and page["error"] is None
    assert page["texts"] == ["Let me check the weather.", "It is sunny and 21C in Paris."]
    assert page["tools"] == [
        {"id": "call_paris", "state": "output-available", "input": {"city": "Paris"}, "output": "Sunny, 21C in Paris"}
    ]
