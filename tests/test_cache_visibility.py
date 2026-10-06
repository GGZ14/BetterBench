"""A prefix-cache hit must be visible in the result.

A lane that keeps a prefix cache warm across runs — and one that persists it to
disk, where it survives a restart — answers a repeated prompt from cache. The
result is a legitimate-looking number that measured a cache lookup. The server
already reports how much it reused (`usage.prompt_tokens_details.cached_tokens`);
this records it, so a warm run is distinguishable from a cold one after the fact
rather than only in hindsight.

Measured on a live RDNA4 lane with a persistent disk prefix tier: a replayed
48k-token prompt went 6.72 s cold -> 0.43 s warm, `cached_tokens` 47104.
"""
from __future__ import annotations

from betterbench.client import RunResult, _Timeline, _finalize
from betterbench.report import cache_hits


def _finalized(usage):
    tl = _Timeline()
    tl.times = [0.0, 0.01]
    return _finalize(RunResult(ok=True), tl, 0.0, 0.02, usage)


def test_cached_tokens_is_recorded_from_usage_details():
    r = _finalized({"prompt_tokens": 48063, "completion_tokens": 2,
                    "prompt_tokens_details": {"cached_tokens": 47104}})
    assert r.cached_tokens == 47104


def test_a_server_that_reports_nothing_leaves_it_none():
    """None is "not reported", never 0 — a 0 would claim a verified cold cache."""
    assert _finalized({"prompt_tokens": 10, "completion_tokens": 2}).cached_tokens is None


def test_report_counts_hits_across_phases():
    results = {"single_stream": {"code": [{"cached_tokens": 5}, {"cached_tokens": 0}]},
               "prefill": [{"cached_tokens": [0, 47104]}],
               "concurrency": [{"cached_tokens": []}]}
    assert cache_hits(results) == 2
    assert cache_hits({}) == 0


def test_html_header_carries_the_warning_too():
    """The HTML report is the artifact people screenshot; the markdown warning
    alone would miss it."""
    from betterbench.html_report import _header
    html = _header({"corpus_version": "1",
                    "single_stream": {"code": [{"cached_tokens": 9}]}},
                   {"unique_nonce": True}, {})
    assert "prefix cache" in html and "not cold" in html
    assert "not cold" not in _header({"single_stream": {}}, {"unique_nonce": True}, {})


def test_markdown_header_warns_only_on_a_hit():
    from betterbench.report import render_markdown
    base = {"corpus_version": "1", "config": {"unique_nonce": True, "temperature": 0.7}}
    warm = render_markdown({**base, "single_stream": {"code": [{"cached_tokens": 9}]}})
    assert "not cold" in warm
    assert "not cold" not in render_markdown({**base, "single_stream": {"code": [{"cached_tokens": 0}]}})
