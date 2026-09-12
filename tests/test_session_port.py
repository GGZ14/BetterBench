"""The interactive compare session: a stdlib HTTP server bound to
127.0.0.1 on the kernel-assigned port 0, the three routes over a re-scanned
runs dir, and the `compare` no-args CLI branch that opens it.

The `_make_run` fixture pattern is copied from `tests/test_gallery.py` (test
files in this repo don't import each other).
"""
from __future__ import annotations

import http.client
import json
from pathlib import Path

import numpy as np
import pytest

from betterbench import session
from betterbench.cli import main
from betterbench.metrics import paired_compare
from betterbench.runs import betterbench_home


def _make_run(root: Path, slug: str, *, model, endpoint, greedy,
              n_passes=4):
    """Write `root/slug/results.json` in a minimal schema-2 shape and
    return the results dict."""
    d = root / slug
    d.mkdir(parents=True)
    recs = [{"ok": True, "category": "prose", "ttft_ms": 48.0,
             "decode_tps": 12.5, "update_gaps_ms": [20.0, 21.0],
             "completion_tokens": 100, "n_chunks": 100,
             "chunking": "per_token", "finish_reason": "stop"}
            for _ in range(n_passes)]
    results = {
        "schema": 2, "corpus_version": "1.0",
        "betterbench_version": "0.9.9",
        "env": {"timestamp": "2026-09-11T08:32:23+0200", "model": model,
                "endpoint": endpoint, "host": "bench-host"},
        "config": {"greedy": greedy, "temperature": 0.7,
                   "runs_per_category": n_passes,
                   "weights": {"prose": 0.2, "code": 0.2, "math": 0.2,
                               "json": 0.2, "reasoning": 0.1, "chat": 0.1}},
        "single_stream": {"prose": [dict(r) for r in recs]},
    }
    (d / "results.json").write_text(json.dumps(results))
    return results


def _get(port: int, path: str) -> tuple[int, str]:
    """One raw GET: `path` goes on the wire verbatim (so a percent-
    encoding in it is never double-encoded by a client)."""
    c = http.client.HTTPConnection("127.0.0.1", port)
    c.request("GET", path)
    r = c.getresponse()
    body = r.read().decode("utf-8")
    c.close()
    return r.status, body


def test_zero_runs_exits_with_message(tmp_path):
    with pytest.raises(SystemExit) as e:
        session.start(tmp_path)
    msg = str(e.value)
    assert "no reportable runs" in msg
    assert "betterbench run" in msg


def test_cli_one_or_three_args_errors():
    with pytest.raises(SystemExit) as e:
        main(["compare", "a.json"])
    assert "expected 2 results files" in str(e.value)
    with pytest.raises(SystemExit) as e:
        main(["compare", "a", "b", "c"])
    assert "expected 2 results files" in str(e.value)


def test_routes_serve_gallery_run_and_pair(tmp_path):
    _make_run(tmp_path, "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    _make_run(tmp_path, "20260114T000000-b", model="mb",
             endpoint="http://b:1", greedy=False)
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        code, body = _get(port, "/")
        assert code == 200
        assert "20260101T000000-a" in body
        assert "20260114T000000-b" in body

        code, body = _get(port, "/run/20260101T000000-a")
        assert code == 200
        assert body.startswith("<!doctype html>")
        assert "← All runs" in body            # the session bar link
        assert "Compare…" in body             # the session bar select label
        assert 'value="20260114T000000-b"' in body   # the other run's option
        assert '"use strict"' in body        # the chart script survived

        code, body = _get(port,
                          "/pair?a=20260101T000000-a&b=20260114T000000-b")
        assert code == 200
        assert '/run/20260101T000000-a' in body    # iframe A
        assert '/run/20260114T000000-b' in body    # iframe B
        assert "cross-file compare is unpaired in time" in body
    finally:
        srv.server_close()


def test_unknown_slugs_are_404(tmp_path):
    _make_run(tmp_path, "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        code, body = _get(port, "/pair?a=20260101T000000-a&b=nope")
        assert code == 404
        assert "no such run: nope" in body

        code, body = _get(port, "/run/nope")
        assert code == 404
        assert "no such run: nope" in body

        # Raw percent-encoded traversal: `%2e%2e` decodes to `..`, which
        # fails the slug validation, so it can never resolve to a file.
        code, _ = _get(port, "/%2e%2e/%2e%2e/etc/passwd")
        assert code == 404

        code, body = _get(port, "/nope")
        assert code == 404
        assert body == "not found\n"
    finally:
        srv.server_close()


def test_missing_pair_param_is_plain_404(tmp_path):
    _make_run(tmp_path, "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        code, body = _get(port, "/pair")
        assert code == 404
        assert body == "not found\n"

        code, body = _get(port, "/pair?a=20260101T000000-a")
        assert code == 404
        assert body == "not found\n"
    finally:
        srv.server_close()


def test_skipped_run_is_404_with_explanation(tmp_path):
    ab = tmp_path / "ab-only-2026"
    ab.mkdir()
    (ab / "ab.json").write_text(json.dumps({"model": "m", "pairs": 0}))
    bad = tmp_path / "corrupt-2026"
    bad.mkdir()
    (bad / "results.json").write_text("{not json")
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        code, body = _get(port, "/run/ab-only-2026")
        assert code == 404
        assert "no results.json" in body      # the RunEntry.error

        code, body = _get(port, "/run/corrupt-2026")
        assert code == 404
        try:
            json.loads("{not json")
        except json.JSONDecodeError as je:
            assert str(je) in body           # the entry's error, verbatim
    finally:
        srv.server_close()


def test_percent_named_run_round_trips_all_routes(tmp_path):
    """A run whose directory name contains a literal `%` round-trips
    through every route under the encode-once/decode-once convention:
    the links are `quote(slug, safe="")` on the wire and the handler
    decodes exactly once, so the name matches the listing."""
    import urllib.parse
    pct = "a%20b"                     # a dir literally named a-%-2-0-b
    _make_run(tmp_path, pct, model="mp", endpoint="http://p:1",
              greedy=True)
    _make_run(tmp_path, "20260101T000000-a", model="ma",
              endpoint="http://a:1", greedy=True)
    enc = urllib.parse.quote(pct, safe="")       # 'a%2520b'
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        # (a) the gallery link is the once-encoded slug
        code, body = _get(port, "/")
        assert code == 200
        assert f'href="/run/{enc}"' in body

        # (b) the once-encoded path decodes to the literal name
        code, body = _get(port, f"/run/{enc}")
        assert code == 200
        assert "← All runs" in body

        # (c) the pair route (parse_qs decodes once, nothing more)
        code, body = _get(port,
                          f"/pair?a={enc}&b=20260101T000000-a")
        assert code == 200
        # (d) the pair's iframe src is the once-encoded slug too
        assert f'src="/run/{enc}"' in body
    finally:
        srv.server_close()


def test_next_run_is_visible_on_the_next_request(tmp_path):
    _make_run(tmp_path, "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    srv = session.make_server(tmp_path)
    port = srv.server_address[1]
    try:
        _code, body = _get(port, "/")
        assert "20260101T000000-a" in body
        assert "20260115T000000-b" not in body
        # A run lands on disk while the session is open:
        _make_run(tmp_path, "20260115T000000-b", model="mb",
                 endpoint="http://b:1", greedy=False)
        _code, body = _get(port, "/")        # re-scan: no cache anywhere
        assert "20260115T000000-b" in body
    finally:
        srv.server_close()


def test_compare_two_arg_output_pinned(tmp_path, capsys):
    """The two-arg `compare` terminal output, pinned byte for byte: an
    argparse regression (e.g. `results` going back to positional
    `a`/`b`) breaks this test, and any change to the table's shape does
    too. The medians below are recomputed here from the same series the
    files carry, exactly as `cmd_compare` computes them."""
    a = tmp_path / "a.json"
    b = tmp_path / "b.json"
    a_data = [12.0, 12.1, 11.9, 12.0]
    b_data = [12.6, 12.7, 12.5, 12.6]

    def one(vals) -> dict:
        recs = [{"ok": True, "category": "prose", "ttft_ms": 48.0,
                 "decode_tps": v, "update_gaps_ms": [20.0, 21.0],
                 "completion_tokens": 100, "n_chunks": 100,
                 "chunking": "per_token", "finish_reason": "stop"}
                for v in vals]
        return {"schema": 2, "single_stream": {"prose": recs}}

    a.write_text(json.dumps(one(a_data)))
    b.write_text(json.dumps(one(b_data)))

    main(["compare", str(a), str(b)])
    out = capsys.readouterr().out

    pr = paired_compare(a_data, b_data, "prose", higher_is_better=True)
    row = (f"| prose | {np.median(a_data):.1f} | {np.median(b_data):.1f} | "
           f"{pr.pct_diff:+.2f}% | [{pr.ci_low_pct:+.1f}%,{pr.ci_high_pct:+.1f}%] | "
           f"{'SIG' if pr.significant else 'noise'} |")
    expected = ("# BetterBench compare (offline, per-category decode t/s)\n"
                "\n"
                "| category | A med | B med | Δ% | 95% CI | verdict |\n"
                "|---|--:|--:|--:|---|---|\n"
                + row + "\n"
                "\n"
                "*Cross-file compares are unpaired in time; prefer "
                "`betterbench ab` for interleaved, drift-cancelled "
                "comparisons.*\n")
    assert out == expected
    assert "# BetterBench compare (offline, per-category decode t/s)" in out
    assert "| category | A med | B med | Δ% | 95% CI | verdict |" in out
    assert f"| prose | {np.median(a_data):.1f}" in out
    assert f"| {np.median(b_data):.1f}" in out
    assert "Cross-file compares are unpaired in time" in out


def test_two_consecutive_sessions_get_different_ports(tmp_path):
    _make_run(tmp_path, "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    s1 = session.make_server(tmp_path)
    try:
        s2 = session.make_server(tmp_path)      # both open concurrently
        try:
            assert s1.server_address[1] != s2.server_address[1]
        finally:
            s2.server_close()
    finally:
        s1.server_close()


def test_cli_no_args_routes_to_session(tmp_path, monkeypatch):
    monkeypatch.setenv("BETTERBENCH_HOME", str(tmp_path))
    (tmp_path / "runs").mkdir()
    _make_run(tmp_path / "runs", "20260101T000000-a", model="ma",
             endpoint="http://a:1", greedy=True)
    import betterbench.session as session_mod
    calls: list = []
    monkeypatch.setattr(session_mod, "start",
                        lambda d: calls.append(d))
    main(["compare"])
    assert len(calls) == 1
    assert str(calls[0]).endswith("/runs")
    assert str(betterbench_home()) == str(tmp_path)
