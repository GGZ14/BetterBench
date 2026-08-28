"""The A/B latency row printed its delta with the opposite sign convention from
the throughput row directly above it, and its CI bounds back to front."""
from __future__ import annotations

from betterbench.metrics import paired_compare
from betterbench.report import render_ab_markdown


def test_lower_latency_reads_as_a_win_with_ordered_bounds():
    r = paired_compare([10.0] * 8, [8.0] * 8, "gap_median_ms",
                       higher_is_better=False)
    assert r.pct_diff < 0                      # B is 20% quicker
    assert r.ci_low_pct <= r.ci_high_pct       # not swapped
    assert "faster" in r.verdict


def test_higher_latency_reads_as_a_loss():
    r = paired_compare([8.0] * 8, [10.0] * 8, "gap_median_ms",
                       higher_is_better=False)
    assert r.pct_diff > 0
    assert r.ci_low_pct <= r.ci_high_pct
    assert "slower" in r.verdict


def test_direction_survives_serialisation():
    d = paired_compare([10.0] * 8, [8.0] * 8, "m", higher_is_better=False).as_dict()
    assert d["higher_is_better"] is False


def test_ab_markdown_uses_the_configured_mde_and_confidence():
    ab = {
        "endpoint_a": "a", "endpoint_b": "b", "model": "m", "pairs": 30,
        "paired_cv": 0.01, "pairs_needed_for_mde": 9,
        "target_mde_pct": 0.5, "conf": 0.99,
        "decode_tps": paired_compare([100.0] * 8, [101.0] * 8, "decode_tps",
                                     conf=0.99).as_dict(),
        "gap_median": {**paired_compare([10.0] * 8, [9.0] * 8, "gap_median_ms",
                                        conf=0.99, higher_is_better=False).as_dict(),
                       "batched": True},
    }
    md = render_ab_markdown(ab)
    assert "0.5% MDE" in md          # was hardcoded to 1.0
    assert "99% CI" in md            # was hardcoded to 95%
    assert "lower is better" in md
    assert "Stream-update gap" in md
