"""The finding: a per-token latency must not be synthesised from a batched chunk.

Each test here pins one shape of stream that a real server produces.
"""
from __future__ import annotations

import statistics

import pytest

from betterbench.client import classify_chunking, stream_chat_sync

MSGS = [{"role": "user", "content": "hello"}]


def _run(base, **kw):
    r = stream_chat_sync(base, "mock", MSGS, max_tokens=kw.pop("max_tokens", 40),
                         temperature=0.0)
    assert r.ok, r.error
    return r


def test_one_token_per_chunk_gap_is_the_itl(server):
    base = server(ttft_ms=5.0, itl_ms=10.0, tokens=40)
    r = _run(base)
    assert r.chunking == "per_token"
    assert r.itl_ms == r.update_gaps_ms          # the gap *is* the ITL, unscaled
    assert len(r.update_gaps_ms) == r.n_chunks - 1
    assert 9.0 <= statistics.median(r.itl_ms) < 16.0


def test_batched_stream_reports_no_per_token_itl(server):
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40, tokens_per_chunk=4)
    r = _run(base)
    assert r.chunking == "batched"
    assert r.tokens_per_update == pytest.approx(4.0, rel=0.05)
    # The tokens in an update arrived together. There is no time between them.
    assert r.itl_ms == []
    assert len(r.update_gaps_ms) == r.n_chunks - 1


def test_a_real_stall_survives_at_full_size(server):
    """The regression guard for the bug itself.

    A 300 ms stall on a 4-tokens-per-update stream was divided by ~4 and
    reported as ~75 ms of per-token latency, hiding the stutter a user feels.
    """
    base = server(ttft_ms=5.0, itl_ms=2.0, tokens=40, tokens_per_chunk=4,
                  stall_every=3, stall_ms=300.0)
    r = _run(base)
    assert r.chunking == "batched"
    assert max(r.update_gaps_ms) >= 290.0
    assert r.itl_ms == []


def test_eos_off_by_one_is_not_batching(server):
    """usage counts the stop token; the stream is still one token per update."""
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40, usage_extra_tokens=1)
    r = _run(base)
    assert r.chunking == "per_token"
    assert r.itl_ms == r.update_gaps_ms


def test_short_run_with_eos_is_not_batching(server):
    """A ratio-only test fails here: 16/15 is 6.7% off but still 1:1."""
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=15, usage_extra_tokens=1)
    r = _run(base, max_tokens=15)
    assert r.chunking == "per_token"


def test_missing_usage_is_unknown_not_assumed(server):
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=40, no_usage=True)
    r = _run(base)
    assert r.chunking == "unknown"
    assert r.tokens_per_update is None
    assert r.itl_ms == []                 # we do not guess a ratio we cannot see
    assert r.update_gaps_ms                # ...but the measurement still stands
    assert r.decode_tps is not None        # and throughput still reports


def test_itl_stays_out_of_the_serialised_record(server):
    """itl_ms is derived, so results.json carries the measured series once."""
    base = server(ttft_ms=5.0, itl_ms=5.0, tokens=20)
    d = _run(base, max_tokens=20).as_dict()
    assert "update_gaps_ms" in d
    assert "itl_ms" not in d


@pytest.mark.parametrize("comp,n,expect", [
    (120, 120, "per_token"),      # exact
    (132, 131, "per_token"),      # +1 EOS, long run
    (16, 15, "per_token"),        # +1 EOS, short run
    (900, 200, "batched"),        # speculative decoding
    (None, 50, "unknown"),        # no usage
    (50, 60, "unknown"),          # fewer tokens than chunks: usage is unreliable
])
def test_classification_table(comp, n, expect):
    assert classify_chunking(comp, n)[0] == expect
