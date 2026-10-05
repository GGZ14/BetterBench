"""The concurrency sweep must actually run the level it claims to.

Blocking streams run one thread each (`client.stream_chat` -> `asyncio.to_thread`),
so the loop's default executor caps how many can be in flight. Left at asyncio's
default — `min(32, cpu+4)`, as low as 5 on a small box — a "c16" level quietly
serialises and the sweep times its own thread pool instead of the server: the
aggregate stops growing at the executor size and reads as a server-side knee.

The assertion is a barrier: N requests that each block until all N have started.
If the executor cannot carry the level, the barrier times out — the test fails
fast, it cannot pass slowly.
"""
from __future__ import annotations

import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

from betterbench.config import Config
from betterbench.runner import concurrency_sweep

N = 24


def _result():
    return SimpleNamespace(ok=True, completion_tokens=1, ttft_ms=1.0,
                           decode_tps=1.0, tokens_per_update=1.0,
                           chunk_token_mismatch=False)


def test_a_sweep_level_starts_every_request_at_once():
    barrier = threading.Barrier(N, timeout=10)

    def blocking_request():
        barrier.wait()          # BrokenBarrierError if the level serialises
        return _result()

    async def request(*args, **kw):
        return await asyncio.to_thread(blocking_request)

    async def go():
        with patch("betterbench.runner._one", request):
            return await concurrency_sweep(
                "unused", "unused", {"test": [None] * N},
                Config(concurrency_levels=[N], concurrency_requests=N))

    out = asyncio.run(go())
    assert out[0]["ok"] == N


def test_a_small_executor_really_does_cap_blocking_streams():
    """Characterisation of why the sweep has to size its own executor: three
    requests waiting on each other never all start when the executor holds two
    workers, so a sweep left on the default would report a level it never ran."""
    tripped = []

    def block(barrier):
        try:
            barrier.wait(1)
            tripped.append(1)
        except threading.BrokenBarrierError:
            pass
        return True

    async def go():
        barrier = threading.Barrier(3)
        await asyncio.gather(*(asyncio.to_thread(block, barrier) for _ in range(3)))
        return len(tripped)

    with ThreadPoolExecutor(max_workers=2) as ex:
        async def main():
            asyncio.get_running_loop().set_default_executor(ex)
            return await go()
        started = asyncio.run(main())
    assert started == 0, "the 3rd request never got a worker"
