from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ._protocol import Scope
from ._transport import AsyncTransport, Connector, SyncTransport
from ._writer import Settings

#: The server checks a ``pub`` token once, when the request starts.
PUB_TOKEN_TTL = 300


@dataclass
class Core:
    """What a publisher takes from its client. The two transport factories are where tests put a fake."""

    settings: Settings
    connector: Connector
    mint: Callable[[Scope, str, float], str]
    sync_transport: Callable[[], SyncTransport]
    async_transport: Callable[[], AsyncTransport]
