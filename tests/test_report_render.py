"""Both table shapes render, and the batched one never shows a per-token number."""
from __future__ import annotations

import json
from pathlib import Path

from betterbench.html_report import render_html
from betterbench.report import render_markdown

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests" / "data" / "report_golden.html"


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


def _wide_results():
    """Single-stream + concurrency + prefill in one fixture; drives every
    section builder, both table shapes' inputs, and every chart."""
    a = _results(4.0)   # batched
    b = _results(1.0)   # per-token
    return {
        "schema": 2,
        "corpus_version": "1.0",
        "env": {
            "model": "fixture-model", "endpoint": "http://127.0.0.1:0/v1",
            "host": "fixture-host", "timestamp": "2025-01-01T00:00:00",
            "gpu": {"nvidia_smi": "NVIDIA Fixture Card"},
            "max_model_len": 8192,
            "corpus_hash": "abcdef0123",
            "notes": {"kernel": "fixture"},
        },
        "config": {"greedy": True, "runs_per_category": 3, "unique_nonce": True,
                   "weights": {"prose": 0.5, "code": 0.5}},
        "single_stream": {
            "prose": a["single_stream"]["prose"],
            "code": b["single_stream"]["prose"],
        },
        "concurrency": [
            {"level": 1, "ok": 5, "requests": 5, "aggregate_tps": 120.5,
             "ttft_ms": [40.0, 41.0, 39.0], "decode_tps": [120.5, 119.8, 121.2]},
            {"level": 4, "ok": 4, "requests": 4, "aggregate_tps": 410.25,
             "ttft_ms": [120.0, 118.0, 122.0], "decode_tps": [102.5, 101.9, 103.1]},
            {"level": 8, "ok": 7, "requests": 8, "aggregate_tps": 733.9,
             "ttft_ms": [310.0, 297.0, 324.0], "decode_tps": [91.7, 90.2, 93.5]},
        ],
        "prefill": [
            {"target_depth": 1024, "prompt_tokens": [1023.0, 1021.0],
             "ttft_ms": [9.0, 9.4], "pp_tps": [113.6, 108.7]},
            {"target_depth": 4096, "prompt_tokens": [4091.0, 4087.0],
             "ttft_ms": [38.0, 37.2], "pp_tps": [107.7, 109.9]},
            {"target_depth": 16384, "prompt_tokens": [16379.0, 16371.0],
             "ttft_ms": [151.0, 150.2], "pp_tps": [108.5, 109.0]},
            {"target_depth": 32768, "skipped": True},
        ],
    }


def test_render_html_is_byte_identical_to_golden():
    assert render_html(_wide_results()) == GOLDEN.read_text()


def test_render_sections_and_render_document_compose_to_the_document():
    from betterbench.html_report import (render_document, render_sections)
    fixture = _wide_results()
    sections = render_sections(fixture)
    assert set(sections) == {"title", "header", "tiles", "figures",
                           "tables", "footer", "data"}
    document = render_document(sections)
    assert document == render_html(fixture)
    assert document == GOLDEN.read_text()
    assert isinstance(sections["data"], dict)
    golden = GOLDEN.read_text()
    start = golden.index('"use strict";\nconst D = ')
    end = golden.index(";\nconst NS", start)
    snippet = golden[start + len('"use strict";\nconst D = '):end]
    assert json.dumps(sections["data"]) == snippet
