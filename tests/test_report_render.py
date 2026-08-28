"""Both table shapes render, and the batched one never shows a per-token number."""
from __future__ import annotations

from betterbench.html_report import render_html
from betterbench.report import render_markdown


def _results(tok_per_update, n=200):
    """A synthetic single-stream result with a known tokens-per-update."""
    chunks = n
    comp = int(n * tok_per_update)
    rec = {"ok": True, "category": "prose", "ttft_ms": 50.0, "decode_tps": 100.0,
           "update_gaps_ms": [20.0] * (chunks - 1), "completion_tokens": comp,
           "n_chunks": chunks, "tokens_per_update": comp / chunks,
           "chunking": "batched" if tok_per_update > 1.1 else "per_token",
           "pp_tps": 1500.0, "finish_reason": "length"}
    return {"schema": 2, "corpus_version": "1.0", "env": {}, "config": {},
            "single_stream": {"prose": [rec]}}


def test_batched_markdown_has_no_itl_columns():
    md = render_markdown(_results(4.0))
    assert "update p50 (ms)" in md and "tok/update" in md
    assert "ITL 1% low" not in md
    assert "1 of 1 runs streamed several tokens per update" in md


def test_per_token_markdown_keeps_the_itl_columns():
    md = render_markdown(_results(1.0))
    assert "ITL 1% low" in md
    assert "update p50 (ms)" not in md


def test_batched_html_switches_table_tile_and_chart():
    html = render_html(_results(4.0))
    assert "update p50 (ms)" in html
    assert "ITL 1% low" not in html
    assert "Combined update p99" in html
    assert '"batched": true' in html or '"batched":true' in html
    assert "Stream-update gap by category" in html


def test_per_token_html_is_unchanged_in_shape():
    html = render_html(_results(1.0))
    assert "ITL 1% low" in html
    assert "Combined ITL 1% low" in html
    assert "Inter-token latency range by category" in html
