"""Shared by both suites. The testing kit's fixtures are opted into here and not in tests/integration/conftest.py:
pytest accepts ``pytest_plugins`` only in a conftest it loads at startup."""

import os

# Agno reports usage to its own servers unless told not to; the tests make no call beyond the local servers.
os.environ.setdefault("AGNO_TELEMETRY", "false")

pytest_plugins = ["iq_stream.testing.pytest_plugin"]
