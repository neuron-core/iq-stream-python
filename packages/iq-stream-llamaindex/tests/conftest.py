"""Shared by both suites. The testing kit's fixtures are opted into here and not in tests/integration/conftest.py:
pytest accepts ``pytest_plugins`` only in a conftest it loads at startup. This directory is on ``sys.path``, so the
suites import ``fakes`` and ``checks`` from it."""

pytest_plugins = ["iq_stream.testing.pytest_plugin"]
