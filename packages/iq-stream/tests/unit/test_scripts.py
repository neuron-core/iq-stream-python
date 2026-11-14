"""The repository's scripts/servers.py starts the short server with its own copy of SHORT_SETTINGS, since it runs
before any package is installed. The two must not drift apart."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from iq_stream.testing import SHORT_SETTINGS

SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "servers.py"


@pytest.mark.skipif(not SCRIPT.exists(), reason="not in a checkout of the repository")
def test_scripts_servers_py_starts_the_short_server_with_the_kits_short_settings() -> None:
    spec = importlib.util.spec_from_file_location("servers_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    assert dict(SHORT_SETTINGS) == script.SHORT_SETTINGS
