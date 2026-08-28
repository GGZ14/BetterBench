"""A v0.2.3 (schema 1) results file must still render — and render honestly.

0.2.3 stored `itl_ms = gap * n_chunks / completion_tokens`, one scalar per run,
so the measured gaps are exactly recoverable and old archives do not have to be
re-run to see what they actually measured.
"""
from __future__ import annotations

import pytest

from betterbench.report import (render_markdown, report_is_batched, run_gaps_ms,
                                run_chunking, single_rows)


def _legacy_record(gaps, comp, n_chunks, **kw):
    """Build a record the way 0.2.3 would have written it."""
    scale = n_chunks / comp
    rec = {"ok": True, "category": "prose", "ttft_ms": 50.0,
           "itl_ms": [g * scale for g in gaps],
           "decode_tps": 100.0, "completion_tokens": comp, "n_chunks": n_chunks,
           "chunk_token_mismatch": abs(comp - n_chunks) / comp > 0.10,
           "finish_reason": "length"}
    rec.update(kw)
    return rec


def test_recovers_the_measured_gap_exactly():
    gaps = [20.0, 25.0, 1296.72, 18.0]      # the stall the old report divided away
    rec = _legacy_record(gaps, comp=390, n_chunks=100)
    assert run_gaps_ms(rec) == pytest.approx(gaps)
    assert max(rec["itl_ms"]) == pytest.approx(1296.72 * 100 / 390)   # was ~332 ms


def test_legacy_batched_file_renders_update_columns():
    recs = [_legacy_record([20.0, 25.0, 900.0], comp=390, n_chunks=100)]
    results = {"schema": 1, "single_stream": {"prose": recs}, "config": {}}
    rows = single_rows(results)
    assert report_is_batched(rows)
    assert rows[0]["itl_med"] is None
    assert rows[0]["tok_per_update"] == pytest.approx(3.9)
    md = render_markdown(results)
    assert "update p99 (ms)" in md
    assert "ITL 1% low" not in md


def test_legacy_non_batched_file_still_renders_itl():
    """The +1-EOS shape of a real non-speculative run."""
    recs = [_legacy_record([10.0] * 9, comp=11, n_chunks=10)]
    results = {"schema": 1, "single_stream": {"prose": recs}, "config": {}}
    rows = single_rows(results)
    assert not report_is_batched(rows)
    assert rows[0]["itl_med"] is not None
    assert "ITL 1% low" in render_markdown(results)


def test_stored_mismatch_flag_does_not_override_the_counts():
    """0.2.3 flagged at 10%; a 5%-off record it left unflagged is still 1:1 —
    and a record it flagged is re-judged by the counts, not by the old flag."""
    rec = _legacy_record([10.0] * 9, comp=11, n_chunks=10)
    rec["chunk_token_mismatch"] = True          # wrong, and deliberately so
    assert run_chunking(rec) == "per_token"
