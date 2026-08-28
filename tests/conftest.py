"""Shared fixtures: a mock OpenAI-compatible server on an ephemeral port.

Binding to port 0 and reading the port back avoids the fixed-port collision
baked into tools/self_test.py, so the suite is safe to run in parallel or on a
box that is already serving a real endpoint.
"""
from __future__ import annotations

import sys
import threading
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

from mock_server import make_handler  # noqa: E402


@pytest.fixture
def server():
    """Factory: `server(ttft_ms=..., itl_ms=..., tokens=..., **knobs) -> base_url`."""
    started = []

    def start(ttft_ms=5.0, itl_ms=5.0, tokens=40, max_ctx=0, **kw):
        srv = ThreadingHTTPServer(("127.0.0.1", 0),
                                  make_handler(ttft_ms, itl_ms, tokens, max_ctx, **kw))
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        started.append((srv, t))
        return f"http://127.0.0.1:{srv.server_address[1]}/v1"

    yield start
    for srv, t in started:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)
