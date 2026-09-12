"""The compare pair page: mismatch chips (two result dicts compared), the
comparison band (paired decode vs unpaired medians), the toolbar, and the
two embedded reports.

Everything is pure over two results dicts and a runs dir; degradation
messages are asserted here (an absent phase renders a muted note, never a
bare number).
"""
from __future__ import annotations

import json
import re

from betterbench.gallery import compare_band, mismatch_chips, render_pair_page
from betterbench.metrics import paired_compare

BANNER = ("cross-file compare is unpaired in time; for a drift-cancelled "
          "verdict run `betterbench ab`. Paired decode stats below use "
          "pass-index pairing truncated to the shorter series.")


def _make_run(root, slug, *, model, endpoint, greedy, temperature=0.7,
             corpus="1.0", host="bench-host", notes=None, gpu=None,
             categories=None, with_concurrency=False, with_prefill=False):
    """Write `root/slug/results.json` in a minimal schema-2 shape and
    return `(path, results dict)`.

    `categories` maps a name -> a list of decode_tps values, or a
    `(value, n_passes)` uniform pair."""
    d = root / slug
    d.mkdir(parents=True)
    cats = categories or {"prose": [12.5] * 4}
    single = {}
    for cat, spec in cats.items():
        vals = ([float(spec[0])] * spec[1] if isinstance(spec, tuple)
                else [float(v) for v in spec])
        recs = [{"ok": True, "category": cat, "ttft_ms": 48.0,
                 "decode_tps": t, "update_gaps_ms": [20.0, 21.0],
                 "completion_tokens": 100, "n_chunks": 100,
                 "chunking": "per_token", "finish_reason": "stop"}
                for t in vals]
        single[cat] = recs
    results = {
        "schema": 2, "corpus_version": corpus, "betterbench_version": "0.9.9",
        "env": {"timestamp": "2026-09-11T08:32:23+0200", "model": model,
                "endpoint": endpoint, "host": host,
                "gpu": (gpu if gpu is not None
                        else {"nvidia_smi": "NVIDIA A100 80GB"}),
                "notes": notes or {}},
        "config": {"greedy": greedy, "temperature": temperature,
                   "runs_per_category": 4,
                   "weights": {"prose": 0.2, "code": 0.2, "math": 0.2,
                               "json": 0.2, "reasoning": 0.1, "chat": 0.1}},
        "single_stream": single,
    }
    if with_concurrency:
        results["concurrency"] = [
            {"level": 1, "ok": 2, "requests": 2, "aggregate_tps": 100.0,
             "ttft_ms": [30.0, 31.0], "decode_tps": [50.0, 51.0]},
            {"level": 8, "ok": 8, "requests": 8, "aggregate_tps": 420.0,
             "ttft_ms": [210.0, 224.0], "decode_tps": [50.0, 52.0]},
        ]
    if with_prefill:
        results["prefill"] = [
            {"target_depth": 200, "skipped": False,
             "prompt_tokens": [150, 150], "ttft_ms": [100.0, 101.0],
             "pp_tps": [1500.0, 1550.0]},
        ]
    (d / "results.json").write_text(json.dumps(results))
    return d, results


def test_mismatch_chips_flags_corpus_sampling_host_notes(tmp_path):
    _, a = _make_run(tmp_path, "r-a", model="m", endpoint="e",
                     greedy=True, corpus="1.0", host="node-a",
                     notes={"driver": "550.54", "rdma": "RoCEv2"},
                     gpu={"nvidia_smi": ["driver header", "NVIDIA A100"]})
    _, b = _make_run(tmp_path, "r-b", model="m", endpoint="e",
                     greedy=False, corpus="1.1", host="node-b",
                     notes={"driver": "552.12"},
                     gpu={"nvidia_smi": "NVIDIA H100"})
    chips = mismatch_chips(a, b)
    assert "corpus v1.0 vs v1.1" in chips
    assert "greedy vs temp 0.7" in chips
    assert "node-a vs node-b" in chips
    assert "NVIDIA A100 vs NVIDIA H100" in chips       # GPU-label family
    assert any(c.startswith("driver:") for c in chips)  # a note value differs
    assert any(c.startswith("rdma:") for c in chips)   # a note on one side only


def test_mismatch_chips_empty_when_identical(tmp_path):
    _, a = _make_run(tmp_path, "r-a", model="m", endpoint="e", greedy=True)
    _, b = _make_run(tmp_path, "r-b", model="m", endpoint="e", greedy=True)
    assert mismatch_chips(a, b) == []
    # total over any pair of result dicts: no keys, no crash
    assert mismatch_chips({}, {}) == []
    assert mismatch_chips({"env": None}, {}) == []
    assert mismatch_chips({"config": None}, {}) == []


def test_compare_band_decodes_paired_stats_and_missing_categories(tmp_path):
    _, a = _make_run(tmp_path, "r-a", model="m", endpoint="e", greedy=True,
                     categories={"prose": (12.0, 5), "math": (20.0, 5)})
    _, b = _make_run(tmp_path, "r-b", model="m", endpoint="e", greedy=True,
                     categories={"prose": (13.0, 3), "code": (9.0, 2)})
    html = compare_band(a, b)
    assert "not measured on B" in html        # math missing from B
    assert "not measured on A" in html        # code missing from A
    assert "+8.33" in html                   # prose: B beats A by 8.33%
    # The verdict is taken from an in-process paired_compare (a zero-
    # variance, non-zero difference is significant), so a future change
    # to the metric code is tracked rather than the SIG/noise rendering
    # asserted here being hardcoded.
    p = paired_compare([12.0] * 5, [13.0] * 3, "prose")
    assert ("SIG" if p.significant else "noise") in html
    # the n<2 path short-circuits to "insufficient pairs", not sig
    p1 = paired_compare([9.0], [9.0], "code")
    assert p1.verdict == "insufficient pairs" and not p1.significant


def test_compare_band_no_shared_stats_shows_muted_note(tmp_path):
    _, a = _make_run(tmp_path, "r-a", model="m", endpoint="e", greedy=True,
                     categories={"math": (20.0, 5)})
    _, b = _make_run(tmp_path, "r-b", model="m", endpoint="e", greedy=True,
                     categories={"code": (9.0, 2)})
    html = compare_band(a, b)
    assert "no decode category measured on both sides" in html
    # only the note, no rows: no per-side cells and no stats
    assert "not measured on A" not in html
    assert "not measured on B" not in html
    assert "SIG" not in html


def test_render_pair_page_wires_toolbar_banner_chips_and_iframes(tmp_path):
    _make_run(tmp_path, "20260114T000000-a", model="m", endpoint="e",
              greedy=True, with_concurrency=True, with_prefill=True)
    _make_run(tmp_path, "20260107T000000-b", model="m", endpoint="e",
              greedy=True, corpus="1.1",  # forced chip
              with_concurrency=True, with_prefill=True)
    _make_run(tmp_path, "20260101T000000-c&d", model="m", endpoint="e",
              greedy=True)
    html = render_pair_page(tmp_path, "20260114T000000-a",
                            "20260107T000000-b")
    # the two embedded reports
    assert 'src="/run/20260114T000000-a"' in html
    assert 'src="/run/20260107T000000-b"' in html
    assert 'title="Report A"' in html and 'title="Report B"' in html
    # the banner sentence, verbatim
    assert BANNER in html
    # the forced mismatch chip
    assert "corpus v1.0 vs v1.1" in html
    # toolbar: each select lists every reportable run except itself
    sel_a = re.search(r'<select id="sel-a">(.*?)</select>', html,
                      re.S).group(1)
    sel_b = re.search(r'<select id="sel-b">(.*?)</select>', html,
                      re.S).group(1)
    # each select excludes its own slug, includes the other and the third
    assert "20260107T000000-b" not in sel_a
    assert "20260114T000000-a" not in sel_b
    assert "20260114T000000-a" in sel_a and "20260107T000000-b" in sel_b
    assert "c%26d" in sel_a and "c%26d" in sel_b
    # a slug with a `&` is URL-encoded (and html-escaped) in option values
    assert 'value="20260101T000000-c%26d"' in sel_a
    assert 'value="20260101T000000-c%26d"' in sel_b
    # and the swap link carries the encoded slugs too
    assert 'href="/pair?a=20260107T000000-b&b=20260114T000000-a"' in html
