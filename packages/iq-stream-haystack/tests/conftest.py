import os

# Haystack sends usage telemetry unless told not to; the tests reach no network beyond the local servers.
os.environ["HAYSTACK_TELEMETRY_ENABLED"] = "False"

pytest_plugins = ["iq_stream.testing.pytest_plugin"]
