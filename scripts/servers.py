#!/usr/bin/env python3
"""Starts and stops the two IQ-Stream servers the integration tests run against, in Docker: one with default
settings on port 8080, one with short timeouts and small limits on port 8081.

    python3 scripts/servers.py start [--ports 8080,8081]
    python3 scripts/servers.py stop

The image is IQ_STREAM_TEST_IMAGE, else ghcr.io/neuron-core/iq-stream:1; the credentials IQ_STREAM_TEST_APP_KEY and
IQ_STREAM_TEST_APP_SECRET, else the defaults of the test kit. The containers' ids are recorded in .servers.json next
to this file, which `stop` reads. Tests find the servers on the two addresses (or on IQ_STREAM_TEST_URL and
IQ_STREAM_TEST_SHORT_URL). The standard library only: CI runs this before any package is installed.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

#: The same as iq_stream.testing.SHORT_SETTINGS; a unit test of the core keeps the two equal.
SHORT_SETTINGS = {
    "PUBLISH_READ_TIMEOUT_MS": "2000",
    "STREAM_IDLE_TIMEOUT_MS": "6000",
    "PENDING_TIMEOUT_MS": "5000",
    "RETRY_MS": "500",
    "RETENTION_MS": "8000",
    "MAX_FRAME_BYTES": "1024",
    "MAX_STREAM_BYTES": "8192",
}
DEFAULT_IMAGE = "ghcr.io/neuron-core/iq-stream:1"
RECORD = Path(__file__).with_name(".servers.json")

# Local servers are reached directly, whatever proxy the environment names.
_opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def docker(*args: str) -> str:
    try:
        completed = subprocess.run(["docker", *args], capture_output=True, text=True, check=True, timeout=300)
    except FileNotFoundError:
        sys.exit("docker is not installed, or not on PATH")
    except subprocess.CalledProcessError as error:
        # Docker's own words, and not the command line, which carries the app secret.
        sys.exit(f"docker {args[0]} failed: {(error.stderr or '').strip() or 'is Docker running?'}")
    return completed.stdout.strip()


def healthy(url: str) -> bool:
    try:
        with _opener.open(f"{url}/healthz", timeout=1) as response:
            return response.status == 200
    except (urllib.error.URLError, OSError):
        return False


def start(ports: tuple[int, int]) -> None:
    if RECORD.exists():
        sys.exit(f"{RECORD} exists: servers started by this script are running. Run `stop` first.")
    for port in ports:
        if healthy(f"http://127.0.0.1:{port}"):
            sys.exit(f"A server already answers on 127.0.0.1:{port}. The tests will use it; nothing was started.")
    image = os.environ.get("IQ_STREAM_TEST_IMAGE") or DEFAULT_IMAGE
    credentials = {
        "APP_KEY": os.environ.get("IQ_STREAM_TEST_APP_KEY") or "demo-app",
        "APP_SECRET": os.environ.get("IQ_STREAM_TEST_APP_SECRET") or "test-app-secret-0123456789abcdef0123",
    }
    containers: list[str] = []
    for port, settings in zip(ports, ({}, SHORT_SETTINGS), strict=True):
        variables = {**credentials, **settings}
        flags = [part for name, value in variables.items() for part in ("-e", f"{name}={value}")]
        containers.append(docker("run", "-d", "-p", f"127.0.0.1:{port}:8080", *flags, image))
        RECORD.write_text(json.dumps(containers))
    for port in ports:
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 30
        while not healthy(url):
            if time.monotonic() > deadline:
                stop()
                sys.exit(f"The server from {image} did not answer on {url}/healthz.")
            time.sleep(0.2)
        print(f"IQ-Stream server on {url}{' (short settings)' if port == ports[1] else ''}")


def stop() -> None:
    if not RECORD.exists():
        return
    containers = json.loads(RECORD.read_text())
    if containers:
        docker("rm", "-f", *containers)
    RECORD.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Start or stop the IQ-Stream servers of the integration tests.")
    parser.add_argument("command", choices=["start", "stop"])
    parser.add_argument("--ports", default="8080,8081", help="the default server's port and the short one's")
    options = parser.parse_args()
    if options.command == "stop":
        stop()
        return
    default, short = (int(port) for port in options.ports.split(","))
    start((default, short))


if __name__ == "__main__":
    main()
