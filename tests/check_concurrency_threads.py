"""Standalone regression: all 24 blocking requests must start before any finish."""
import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import patch
from betterbench.config import Config
from betterbench.runner import concurrency_sweep

barrier = threading.Barrier(24, timeout=5)
def blocking_request():
    barrier.wait()
    return SimpleNamespace(ok=True, completion_tokens=1, ttft_ms=1,
                           decode_tps=1, tokens_per_update=1, chunk_token_mismatch=False)
async def request(*args):
    return await asyncio.to_thread(blocking_request)
with patch('betterbench.runner._one', request):
    result = asyncio.run(concurrency_sweep('unused', 'unused', {'test': [None]},
        Config(concurrency_levels=[24], concurrency_requests=24)))
assert result[0]['ok'] == 24
print('PASS: 24 blocking requests active simultaneously')
