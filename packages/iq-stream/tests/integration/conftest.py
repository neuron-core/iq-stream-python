from __future__ import annotations

import pytest
from helpers import Run

from iq_stream.testing import TestServers


@pytest.fixture(params=["sync", "async"])
def kind(request: pytest.FixtureRequest) -> str:
    """Each scenario runs once for Publisher and once for AsyncPublisher."""
    return str(request.param)


@pytest.fixture
def run(kind: str, iq_servers: TestServers) -> Run:
    return Run(kind, iq_servers)
