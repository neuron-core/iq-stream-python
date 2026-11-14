from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fakes import Harness

from iq_stream import _publisher as publisher_module


@pytest.fixture(params=["sync", "async"])
def kind(request: pytest.FixtureRequest) -> str:
    """Each state-machine test runs once for Publisher and once for AsyncPublisher."""
    return str(request.param)


@pytest.fixture
def setup(kind: str) -> Callable[..., Harness]:
    def make(**options: Any) -> Harness:
        return Harness(kind, **options)

    return make


@pytest.fixture(autouse=True)
def no_writer_left_behind() -> Iterator[None]:
    """Every test leaves no writer thread and no open publisher behind."""
    yield
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        writers = [thread for thread in threading.enumerate() if thread.name.startswith("iq-stream ")]
        if not writers and not publisher_module._open:  # pyright: ignore[reportPrivateUsage]
            return
        time.sleep(0.01)
    raise AssertionError(f"left behind: {writers}, open publishers {publisher_module._open}")  # pyright: ignore[reportPrivateUsage]
