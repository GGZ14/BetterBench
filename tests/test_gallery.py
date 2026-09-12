"""Run scanning, run manifest, and the gallery page — pure functions over a runs dir.

The headline numbers are recomputed through the same row builders the report
(`betterbench.report`) uses, so the gallery can't drift from it.
"""
from __future__ import annotations

import json
from pathlib import Path

from betterbench.gallery import (list_runs, render_gallery, render_pair_page,
                                 run_manifest)


def _make_run(root: Path, slug: str, *, model, endpoint, greedy, corpus="1.0",
              n_passes=4, with_concurrency=False, with_prefill=False,
              broken_json=False, notes=None, max_model_len=None,
              drop_conc_level=False):
    """Write `root/slug/results.json` in a minimal schema-2 shape."""
    d = root / slug
    d.mkdir(parents=True)
    recs = [{"ok": True, "category": "prose", "ttft_ms": 48.0,
             "decode_tps": 12.5, "update_gaps_ms": [20.0, 21.0],
             "completion_tokens": 100, "n_chunks": 100, "chunking": "per_token",
             "finish_reason": "stop"} for _ in range(n_passes)]
    results = {
        "schema": 2, "corpus_version": corpus, "betterbench_version": "0.9.9",
        "env": {"timestamp": "2026-09-11T08:32:23+0200", "model": model,
                "endpoint": endpoint, "host": "bench-host"},
        "config": {"greedy": greedy, "temperature": 0.7,
                   "runs_per_category": n_passes,
                   "weights": {"prose": 0.2, "code": 0.2, "math": 0.2,
                               "json": 0.2, "reasoning": 0.1, "chat": 0.1}},
        "single_stream": {"prose": [dict(r) for r in recs]},
    }
    if with_concurrency:
        results["concurrency"] = [
            {"level": 1, "ok": 2, "requests": 2, "aggregate_tps": 100.0,
             "ttft_ms": [30.0, 31.0], "decode_tps": [50.0, 51.0]},
            {"level": 8, "ok": 8, "requests": 8, "aggregate_tps": 420.0,
             "ttft_ms": [210.0, 224.0], "decode_tps": [50.0, 52.0]},
        ]
        if drop_conc_level:
            del results["concurrency"][0]["level"]
    if with_prefill:
        results["prefill"] = [
            {"target_depth": 200, "skipped": False,
             "prompt_tokens": [150, 150], "ttft_ms": [100.0, 101.0],
             "pp_tps": [1500.0, 1550.0]},
        ]
    if notes is not None:
        results["env"]["notes"] = notes
    if max_model_len is not None:
        results["env"]["max_model_len"] = max_model_len
    (d / "results.json").write_text("{not json" if broken_json
                                    else json.dumps(results))
    return d


def test_list_runs_orders_newest_first_and_flags_corrupt_json(tmp_path):
    _make_run(tmp_path, "20260101T000000-old", model="m-old",
             endpoint="http://e-o", greedy=True)
    _make_run(tmp_path, "20260107T000000-mid", model="m-mid",
             endpoint="http://e-m", greedy=True)
    _make_run(tmp_path, "20260114T000000-new", model="m-new",
             endpoint="http://e-n", greedy=False)
    bad = tmp_path / "bad-run"
    bad.mkdir()
    (bad / "results.json").write_text("{not json")
    reportable, skipped = list_runs(tmp_path)
    assert len(reportable) == 3
    assert len(skipped) == 1
    assert skipped[0].slug == "bad-run"
    assert skipped[0].results is None
    try:
        json.loads("{not json")
    except json.JSONDecodeError as e:
        assert str(e) in skipped[0].error
    assert reportable[0].slug == "20260114T000000-new"   # newest first
    assert reportable[-1].slug == "20260101T000000-old"  # oldest last
    assert reportable[0].results is not None
    assert reportable[0].error is None


def test_list_runs_ignores_non_directories_and_ab_only_dirs(tmp_path):
    _make_run(tmp_path, "20260101T000000-valid", model="m",
             endpoint="http://e", greedy=True)
    (tmp_path / "stray.txt").write_text("not a run")
    ab = tmp_path / "ab-only"
    ab.mkdir()
    (ab / "ab.json").write_text(json.dumps({"model": "m", "pairs": 0}))
    reportable, skipped = list_runs(tmp_path)
    assert [e.slug for e in reportable] == ["20260101T000000-valid"]
    assert [e.slug for e in skipped] == ["ab-only"]
    assert "no results.json" in skipped[0].error
    # A missing runs dir is not an error: the caller decides what to do.
    r, s = list_runs(tmp_path / "does-not-exist")
    assert r == [] and s == []


def test_render_gallery_lists_headline_numbers_and_links(tmp_path):
    _make_run(tmp_path, "20260101T000000-old", model="model-alpha",
             endpoint="http://a:1", greedy=True, with_concurrency=True)
    _make_run(tmp_path, "20260114T000000-new", model="model-beta",
             endpoint="http://b:1", greedy=False, with_prefill=True)
    _make_run(tmp_path, "20260107T000000-bad", model="m-bad",
             endpoint="http://x:1", greedy=True, broken_json=True)
    html = render_gallery(tmp_path)
    assert "20260101T000000-old" in html
    assert "20260114T000000-new" in html
    assert "model-alpha" in html and "model-beta" in html
    assert "12.5" in html           # uniform decode_tps -> combined 12.5
    assert "420.0" in html          # aggregate at the top concurrency level
    assert 'href="/run/20260101T000000-old"' in html
    assert 'href="/run/20260114T000000-new"' in html
    assert 'name="sel"' in html and 'value="20260114T000000-new"' in html
    assert 'class="skip"' in html   # the broken run is visible, not dropped
    assert "20260107T000000-bad" in html
    assert "Compare selected" in html


def test_render_gallery_zero_runs_says_run(tmp_path):
    html = render_gallery(tmp_path / "does-not-exist")
    assert "All runs" in html
    assert "betterbench run" in html
    assert "<input" not in html          # no checkboxes when nothing to compare


def test_malformed_run_without_env_key_keeps_gallery_alive(tmp_path):
    _make_run(tmp_path, "20260114T000000-good", model="m-good",
             endpoint="e", greedy=True)
    odd = tmp_path / "20260107T000000-noenv"
    odd.mkdir()
    (odd / "results.json").write_text(json.dumps({
        "schema": 2, "corpus_version": "1.0",
        "config": {"greedy": True, "temperature": 0.7,
                   "runs_per_category": 4, "weights": {"prose": 1.0}},
        "single_stream": {"prose": [
            {"ok": True, "category": "prose", "ttft_ms": 48.0,
             "decode_tps": 12.5, "update_gaps_ms": [20.0],
             "completion_tokens": 100, "n_chunks": 100,
             "chunking": "per_token", "finish_reason": "stop"}]},
    }))
    reportable, _ = list_runs(tmp_path)   # parseable -> still reportable
    assert {e.slug for e in reportable} == {"20260114T000000-good",
                                          "20260107T000000-noenv"}
    html = render_gallery(tmp_path)       # the page must not die on this run
    assert "<!doctype html>" in html
    assert "m-good" in html and "12.5" in html   # the good run is intact
    assert "20260107T000000-noenv" in html      # odd run stays visible


def test_malformed_run_with_notes_as_list_keeps_gallery_alive(tmp_path):
    _make_run(tmp_path, "20260114T000000-good", model="m-good",
             endpoint="e", greedy=True)
    _make_run(tmp_path, "20260107T000000-notes",
             model="m-notes", endpoint="e", greedy=True,
             notes=["hot", 7])
    html = render_gallery(tmp_path)
    assert "m-good" in html and "m-notes" in html
    assert "render error" not in html


def test_malformed_run_with_non_numeric_max_model_len_keeps_gallery_alive(
        tmp_path):
    _make_run(tmp_path, "20260114T000000-good", model="m-good",
             endpoint="e", greedy=True)
    _make_run(tmp_path, "20260107T000000-ctx",
             model="m-ctx", endpoint="e", greedy=True,
             max_model_len="128k")
    html = render_gallery(tmp_path)
    assert "m-good" in html and "m-ctx" in html
    assert "tok context" not in html   # the chip is omitted, not fabricated
    assert "render error" not in html


def test_malformed_concurrency_entry_degrades_manifest_and_pair_page(tmp_path):
    _make_run(tmp_path, "20260114T000000-a", model="ma", endpoint="ea",
             greedy=True, with_concurrency=True, drop_conc_level=True)
    _make_run(tmp_path, "20260107T000000-b", model="mb", endpoint="eb",
             greedy=True, with_concurrency=True)
    reportable, _ = list_runs(tmp_path)
    a = next(e for e in reportable if e.slug == "20260114T000000-a")
    m = run_manifest(a)                      # must not raise
    assert m["aggregate_top_conc"] is None  # number omitted, not fabricated
    html = render_pair_page(tmp_path, "20260114T000000-a",
                           "20260107T000000-b")
    assert "Concurrency" in html
    assert "not measured on A" in html      # degraded to the muted note


def test_non_object_results_json_is_a_muted_render_error_row(tmp_path):
    _make_run(tmp_path, "20260114T000000-good", model="m-good",
             endpoint="e", greedy=True)
    odd = tmp_path / "20260107T000000-array"
    odd.mkdir()
    (odd / "results.json").write_text(json.dumps([1, 2, 3]))
    reportable, _ = list_runs(tmp_path)
    assert {e.slug for e in reportable} == {"20260114T000000-good",
                                          "20260107T000000-array"}
    html = render_gallery(tmp_path)
    assert "m-good" in html                  # the good run is intact
    assert 'class="skip"' in html           # the odd run is a muted row
    assert "render error" in html
