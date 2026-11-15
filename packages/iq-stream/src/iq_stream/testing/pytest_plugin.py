"""pytest fixtures over the two test servers. Not registered by itself: a suite opts in with

    pytest_plugins = ["iq_stream.testing.pytest_plugin"]

in its top-level ``conftest.py``, ``tests/conftest.py``: pytest refuses ``pytest_plugins`` in a conftest further
down, such as ``tests/integration/conftest.py``, as soon as it collects more than that directory. ``iq_servers``
finds (or starts) the servers once per session; ``iq_client`` and ``iq_short_client`` are clients for each, with
``min_open=0`` so that tests do not wait for the close line.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from .._client import IQClient
from ._servers import TestServers, test_servers


@pytest.fixture(scope="session")
def iq_servers() -> Iterator[TestServers]:
    """The default server and the short one (``SHORT_SETTINGS``), with their credentials."""
    servers = test_servers()
    try:
        yield servers
    finally:
        servers.stop()


@pytest.fixture
def iq_client(iq_servers: TestServers) -> IQClient:
    """A client of the default server."""
    return IQClient(iq_servers.default.url, iq_servers.app_key, iq_servers.app_secret, min_open=0.0)


@pytest.fixture
def iq_short_client(iq_servers: TestServers) -> IQClient:
    """A client of the short server, whose frame limit (1,024 bytes) it knows. Its keepalive is the default 20 s,
    longer than that server's 2 s read timeout, so a silent run there exercises resuming; a test of keepalives makes
    its own client with ``keepalive=0.5``."""
    return IQClient(iq_servers.short.url, iq_servers.app_key, iq_servers.app_secret, min_open=0.0, max_frame_bytes=1024)
