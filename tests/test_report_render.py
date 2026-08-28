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


def _thinking_results(known, unknown, share=0.6):
    recs = []
    for _ in range(known):
        recs.append({"ok": True, "category": "prose", "ttft_ms": 20.0,
                     "ttfa_ms": 800.0, "decode_tps": 100.0,
                     "update_gaps_ms": [10.0] * 99, "completion_tokens": 100,
                     "n_chunks": 100, "reasoning_source": "channel",
                     "reasoning_tokens_est": int(100 * share),
                     "answer_tokens_est": 100 - int(100 * share),
                     "finish_reason": "length"})
    for _ in range(unknown):
        recs.append({"ok": True, "category": "prose", "ttft_ms": 20.0,
                     "decode_tps": 100.0, "update_gaps_ms": [10.0] * 99,
                     "completion_tokens": 100, "n_chunks": 100,
                     "reasoning_source": "unknown", "finish_reason": "length"})
    return {"schema": 2, "env": {}, "config": {}, "single_stream": {"prose": recs}}


def test_split_table_appears_with_the_denominator():
    md = render_markdown(_thinking_results(known=8, unknown=2))
    assert "Reasoning / answer split" in md
    assert "8/10" in md and "60%" in md
    assert "2/10" in md                       # never reached answer


def test_mostly_unknown_category_shows_a_dash_not_a_guess():
    """84% of real runs are truncated, so this is the normal case."""
    md = render_markdown(_thinking_results(known=2, unknown=18))
    assert "Reasoning / answer split" in md   # there *is* evidence, so show it
    assert "2/20" in md
    assert "| — |" in md                      # share and TTFA withheld


def test_no_reasoning_evidence_means_no_extra_noise():
    md = render_markdown(_thinking_results(known=0, unknown=6))
    assert "Reasoning / answer split" not in md
    assert "Stopped at `max_tokens`: **6/6**" in md
