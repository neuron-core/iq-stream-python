"""The testing kit's fixtures, for the integration tests. Here and not in tests/integration/conftest.py: pytest
accepts ``pytest_plugins`` only in a conftest it loads at startup, and ``pytest tests`` loads that one later."""

pytest_plugins = ["iq_stream.testing.pytest_plugin"]
