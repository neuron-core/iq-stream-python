"""The conformance kit for the core and its adapters: a publisher that records in memory, real servers to publish
to, and a subscriber that collects what a stream delivered. It needs nothing beyond the standard library and httpx;
starting a server needs the Docker CLI.

The pytest fixtures are in ``iq_stream.testing.pytest_plugin``, which a test suite opts into with
``pytest_plugins = ["iq_stream.testing.pytest_plugin"]``.
"""

from ._recording import RecordedFrame, RecordedResult, RecordingPublisher
from ._servers import (
    SHORT_SETTINGS,
    ServerAddress,
    StartedServer,
    TestServers,
    free_port,
    healthy,
    start_server,
    test_servers,
)
from ._subscribe import (
    BackgroundSubscription,
    SubscribedFrame,
    Subscription,
    Terminal,
    asubscribe,
    subscribe,
    subscribe_in_background,
)

__all__ = [
    "SHORT_SETTINGS",
    "BackgroundSubscription",
    "RecordedFrame",
    "RecordedResult",
    "RecordingPublisher",
    "ServerAddress",
    "StartedServer",
    "SubscribedFrame",
    "Subscription",
    "Terminal",
    "TestServers",
    "asubscribe",
    "free_port",
    "healthy",
    "start_server",
    "subscribe",
    "subscribe_in_background",
    "test_servers",
]
