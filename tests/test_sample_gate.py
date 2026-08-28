"""`enough_samples_for_percentile` existed, was imported, and was never called —
while METHODOLOGY, DESIGN and both report footers claimed it was. These pin the
claim to the behaviour."""
from __future__ import annotations

from betterbench.report import render_markdown, sample_gate, single_rows


def _results(n_runs, gaps_per_run):
    recs = [{"ok": True, "category": "prose", "ttft_ms": 50.0 + i,
             "decode_tps": 100.0,
             "update_gaps_ms": [10.0] * gaps_per_run,
             "completion_tokens": gaps_per_run + 1, "n_chunks": gaps_per_run + 1,
             "finish_reason": "length"}
            for i in range(n_runs)]
    return {"schema": 2, "env": {}, "config": {}, "single_stream": {"prose": recs}}


def test_ttft_p99_is_flagged_at_any_supported_pass_count():
    """20 passes give n=20 against a required 500. It has never been supported."""
    r = _results(20, 40)
    assert single_rows(r)[0]["ttft_p99_ok"] is False
    assert "†" in render_markdown(r)
    metrics = {u["metric"] for u in sample_gate(r)["under_sampled"]}
    assert "ttft_p99" in metrics


def test_a_token_rich_tail_passes_the_gate():
    r = _results(20, 40)                     # 800 gap samples
    assert single_rows(r)[0]["tail_ok"] is True
    assert {u["metric"] for u in sample_gate(r)["under_sampled"]} == {"ttft_p99"}


def test_a_thin_tail_is_flagged_too():
    """Batching by 4x means 4x fewer gap samples — the new tail metric is
    thinner than the old fabricated one, and must say so."""
    r = _results(5, 60)                      # 300 gap samples, needs 500
    assert single_rows(r)[0]["tail_ok"] is False


def test_gate_records_the_shortfall_not_just_a_boolean():
    entry = next(u for u in sample_gate(_results(20, 40))["under_sampled"]
                 if u["metric"] == "ttft_p99")
    assert entry == {"section": "single_stream", "key": "prose",
                     "metric": "ttft_p99", "n": 20, "need": 500}
