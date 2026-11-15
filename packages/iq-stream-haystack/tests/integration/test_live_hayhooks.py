"""The plan's server integration: the example Hayhooks wrapper under a real Hayhooks server, called over HTTP, its
stream read by a real subscriber."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
from iq_stream.testing import TestServers, free_port, subscribe

from tests.checks import verify_agui
from tests.live import in_order_once

EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "hayhooks"


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
