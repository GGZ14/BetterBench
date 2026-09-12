"""List the runs, summarize each one, and render the all-runs gallery page
and the pair page between two chosen runs.

A *reportable* run is a subdirectory of the runs dir whose `results.json`
parses. Its headline numbers are recomputed through the same row builders
(:mod:`betterbench.report`) the markdown and HTML reports use, so the
gallery can never drift from them. The pair page reuses the A/B command's
stat machinery: a paired-t on the per-pass decode difference
(``paired_compare``, pass-index pairing truncated to the shorter series)
plus unpaired medians for the phases that don't pair without an
interleaved `betterbench ab`. Everything here renders from a directory —
the serving layer runs on top.
"""
from __future__ import annotations

import html
import json
import math
import statistics
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .html_report import _pretty_ts
from .metrics import paired_compare
from .report import (combined_score, concurrency_rows, phases_present,
                     prefill_rows, report_is_batched, single_rows)


@dataclass(frozen=True)
class RunEntry:
    """One subdirectory of the runs directory, classified."""
    slug: str
    path: Path
    results: dict | None      # None when results.json is missing or corrupt
    error: str | None         # why, when not reportable


def list_runs(runs_dir: Path) -> tuple[list[RunEntry], list[RunEntry]]:
    """``(reportable, skipped)`` over the immediate subdirectories of
    ``runs_dir``, slug-descending (name order — 'newest first' for the
    tool's timestamp-prefixed run-dir names).

    An ``ab.json``-only directory is not reportable but never silently
    dropped: it lands in ``skipped`` with the reason, so the page can show
    it. A missing runs directory is not an error — the caller decides.
    """
    entries: list[RunEntry] = []
    if runs_dir.is_dir():
        for d in sorted(runs_dir.iterdir(), key=lambda p: p.name, reverse=True):
            if not d.is_dir():
                continue
            try:
                obj = json.loads((d / "results.json").read_text())
            except FileNotFoundError:
                entries.append(RunEntry(d.name, d, None,
                                       "no results.json — maybe an ab-only run dir"))
            except (json.JSONDecodeError, OSError) as e:
                entries.append(RunEntry(d.name, d, None, str(e)))
            else:
                entries.append(RunEntry(d.name, d, obj, None))
    reportable = [e for e in entries if e.results is not None]
    skipped = [e for e in entries if e.results is None]
    return reportable, skipped


def gpu_label(env: dict) -> str | None:
    """The GPU label, normalized: the first non-empty among the smi keys and
    ``vendor``. The two ``*_smi*`` keys are smi output *lines — lists in
    real files* — so the last non-empty line (the card line) is the label;
    ``vendor`` is a plain string. Missing → ``None`` (chip omitted).
    ``mismatch_chips`` (pair page) compares this same normalized value.
    """
    gpu = env.get("gpu") or {}
    for key in ("nvidia_smi", "rocm_smi_productname"):
        val = gpu.get(key)
        if not val:
            continue
        lines = val if isinstance(val, list) else [val]
        for line in reversed(lines):
            text = str(line).strip()
            if text:
                return text
    v = gpu.get("vendor")
    if v is None:
        return None
    text = str(v).strip()
    return text or None


def _fmt(x, d=1) -> str:
    """A headline number, or a dash. JSON NaN/inf never renders."""
    v = None
    if x is not None:
        try:
            f = float(x)
        except (TypeError, ValueError):
            pass
        else:
            v = f if math.isfinite(f) else None
    if v is None:
        return "—"
    return f"{v:,.{d}f}"


def run_manifest(e: RunEntry) -> dict:
    """Headline numbers for one reportable run, from the report row builders.

    `.get`-safe: must never raise on an odd dict (missing `env`, a
    non-dict `notes`, a non-numeric `max_model_len`, a phase entry the
    row builders can't decode) and must never fabricate numbers — what's
    missing is simply `None` / omitted. The row-builder calls are
    guarded: a builder that can't decode a side's data means that number
    is absent, not an error.

    Keys: name, model, endpoint, timestamp (the report's ``_pretty_ts``
    render, so a run with a colonless ``+0200`` timestamp shows the way its
    report does), chips, phases, combined_decode (weighted; None when there
    are no single-stream rows), ttft_p50 (median of the per-category
    single-stream ttft p50 values), aggregate_top_conc (last concurrency
    level's aggregate t/s), passes_per_category (the run header's own
    number, not guessed from a row).
    """
    if e.results is None:
        raise ValueError(f"run {e.slug!r} has no parseable results — nothing to manifest")
    if not isinstance(e.results, dict):
        raise ValueError(f"run {e.slug!r} results are not an object — nothing to manifest")
    res = e.results
    env = res.get("env") or {}
    if not isinstance(env, dict):
        env = {}
    cfg = res.get("config") or {}
    if not isinstance(cfg, dict):
        cfg = {}
    try:
        rows = single_rows(res)
    except Exception:
        rows = []
    try:
        comb = combined_score(res, rows)
    except Exception:
        comb = None
    try:
        ttfts = [r["ttft_p50"] for r in rows if r.get("ttft_p50") is not None]
    except Exception:
        ttfts = []
    try:
        conc = concurrency_rows(res)
    except Exception:
        conc = []
    chips = [str(env.get("model", "?")),
             str(env.get("endpoint", "?")),
             "corpus v" + str(res.get("corpus_version", "?"))]
    chips.append("greedy" if cfg.get("greedy")
                 else f"temp {cfg.get('temperature')}")
    chips.append("cold prefix cache (nonce)" if cfg.get("unique_nonce")
                 else "warm prefix cache")
    notes = env.get("notes")
    if isinstance(notes, dict):
        for k, v in notes.items():
            chips.append(f"{k}: {v}")
    gl = gpu_label(env)
    if gl:
        chips.append(gl)
    ctx = env.get("max_model_len")
    if ctx:
        try:
            ctx = int(ctx)
        except (TypeError, ValueError):
            ctx = None
        if ctx:
            chips.append(f"{ctx:,} tok context")
    return {
        "name": e.slug,
        "model": env.get("model"),
        "endpoint": env.get("endpoint"),
        "timestamp": _pretty_ts(env.get("timestamp")),
        "chips": chips,
        "phases": phases_present(res),
        "combined_decode": comb["decode"] if comb else None,
        "ttft_p50": statistics.median(ttfts) if ttfts else None,
        "aggregate_top_conc": conc[-1]["aggregate_tps"] if conc else None,
        "passes_per_category": cfg.get("runs_per_category"),
    }


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _or_dash(v) -> str:
    """A value, or the missing marker. JSON null/empty never renders."""
    return "—" if v in (None, "") else str(v)


def _num(v) -> str:
    """A number, or a dash. Unmeasured values never render the bare datum
    (``None`` is the ``not measured`` path, handled by the caller)."""
    return "—" if v is None else f"{v:,.2f}"


def _sampling_label(res: dict) -> str:
    """The sampling chip, read the way the report header reads it: greedy
    truthy vs not, temperature via ``config.get("temperature")``."""
    cfg = res.get("config") or {}
    return "greedy" if cfg.get("greedy") else f"temp {cfg.get('temperature')}"


def mismatch_chips(a: dict, b: dict) -> list[str]:
    """The pair page's comparability band: what differs between the two
    `results.json` files, in this order — corpus version, sampling
    (greedy vs temperature), host, the normalized GPU label (the same
    `gpu_label` `run_manifest` uses), and each `env.notes` key of the
    union of both files whose value differs or is present on only one
    side. Missing sides render as —. `[]` when everything matches.
    Total: any pair of `results.json` dicts returns a list without
    raising (all lookups are `.get`).
    """
    chips: list[str] = []
    ca = a.get("corpus_version")
    cb = b.get("corpus_version")
    if ca != cb:
        chips.append(f"corpus v{ca} vs v{cb}")
    sa = _sampling_label(a)
    sb = _sampling_label(b)
    if sa != sb:
        chips.append(f"{sa} vs {sb}")
    ha = (a.get("env") or {}).get("host")
    hb = (b.get("env") or {}).get("host")
    if ha != hb:
        chips.append(f"{_or_dash(ha)} vs {_or_dash(hb)}")
    ga = gpu_label(a.get("env") or {})
    gb = gpu_label(b.get("env") or {})
    if ga != gb:
        chips.append(f"{_or_dash(ga)} vs {_or_dash(gb)}")
    na = (a.get("env") or {}).get("notes") or {}
    nb = (b.get("env") or {}).get("notes") or {}
    for k in sorted(set(na) | set(nb)):
        va = na.get(k)
        vb = nb.get(k)
        if va != vb:
            chips.append(f"{k}: {_or_dash(va)} vs {_or_dash(vb)}")
    return chips


def _reportable_row(e: RunEntry) -> str:
    m = run_manifest(e)
    chips = "".join(f'<span class="chip">{_esc(c)}</span>' for c in m["chips"])
    val = urllib.parse.quote(str(e.slug), safe="")
    return (f'<tr><td class="slug"><a href="/run/{_esc(val)}">'
            f'{_esc(e.slug)}</a></td>'
            f'<td>{_or_dash(m["model"])}</td>'
            f'<td class="ep">{_or_dash(m["endpoint"])}</td>'
            f'<td class="when">{_esc(m["timestamp"])}</td>'
            f'<td class="chips">{chips}</td>'
            f'<td>{_esc(", ".join(m["phases"]) or "—")}</td>'
            f'<td>{_fmt(m["combined_decode"])}</td>'
            f'<td>{_fmt(m["ttft_p50"])}</td>'
            f'<td>{_fmt(m["aggregate_top_conc"])}</td>'
            f'<td class="pick"><input type="checkbox" name="sel" '
            f'value="{_esc(val)}"></td></tr>')


def _degraded_row(e: RunEntry, exc: Exception) -> str:
    """The muted fallback for one reportable run whose row can't be
    built: the same skip-row look, the exception in the error cell, so
    the bad run stays visible and the rest of the table survives."""
    return (f'<tr class="skip"><td class="slug">{_esc(e.slug)}</td>'
            f'<td colspan="9" class="skiperr">'
            f'{_esc(f"render error: {exc}")}</td></tr>')


def _skipped_row(e: RunEntry) -> str:
    return (f'<tr class="skip"><td class="slug">{_esc(e.slug)}</td>'
            f'<td colspan="9" class="skiperr">skipped — '
            f'{_esc(e.error or "unreadable")}</td></tr>')


_PAGE_CSS = """  :root {
    color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb;
    --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
    --grid:#e1e0d9;
    --s1:#2a78d6; --s2:#eb6834;
    --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      color-scheme: dark;
      --page:#0d0d0d; --surface:#1a1a19;
      --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a;
      --s1:#3987e5; --s2:#d95926;
    }
  }
  * { box-sizing:border-box; }
  body { background:var(--page); color:var(--ink); font-family:var(--sans);
         line-height:1.55; margin:0; padding:32px 20px 72px; }
  .wrap { max-width:1200px; margin:0 auto; display:flex; flex-direction:column; gap:26px; }
  header { display:flex; flex-direction:column; gap:13px; }
  .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.13em;
             text-transform:uppercase; color:var(--muted); }
  h1 { font-size:clamp(26px,4vw,36px); line-height:1.12; margin:0;
       letter-spacing:-.02em; text-wrap:balance; }
  code { font-family:var(--mono); font-size:.95em; }
  .pickbar { display:flex; align-items:center; gap:12px; }
  .pickbar button { font-family:var(--mono); font-size:13px;
                    color:var(--page); background:var(--ink); border:0;
                    border-radius:7px; padding:8px 15px; cursor:pointer; }
  .pickbar button:hover { background:var(--ink-2); }
  .pickbar .hint { font-family:var(--mono); font-size:12px; color:var(--muted); }
  .runs { background:var(--surface); border:1px solid var(--grid);
          border-radius:10px; padding:14px 16px; overflow-x:auto; }
  table { border-collapse:collapse; width:100%; font-size:12.5px;
          font-family:var(--mono); font-variant-numeric:tabular-nums; }
  th, td { padding:7px 12px 7px 0; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  thead th { color:var(--muted); font-weight:500; font-size:10.5px;
             letter-spacing:.07em; text-transform:uppercase;
             border-bottom:1px solid var(--grid); }
  tbody tr + tr td { border-top:1px solid var(--grid); }
  tbody td { color:var(--ink-2); }
  td.slug { color:var(--ink); }
  a { color:var(--s1); text-decoration:none; }
  a:hover { text-decoration:underline; }
  .chips { white-space:normal; max-width:520px; }
  .chip { display:inline-block; margin:1px 4px 1px 0; padding:1px 8px;
          border:1px solid var(--grid); border-radius:6px;
          font-size:11.5px; color:var(--ink-2); background:var(--page); }
  tr.skip td { color:var(--muted); }
  tr.skip td.skiperr { text-align:left; color:var(--muted); }
  tr.skip td.slug { font-style:italic; }
  input[type=checkbox] { accent-color:var(--s1); }
  .empty { max-width:70ch; color:var(--ink-2); }
"""

_SEL_JS = """function compareSel() {
  /* Selection is browser-side only: the first two checked `sel` checkboxes,
     in DOM order, are A and B. Checking more than two uses the first two;
     fewer than two does nothing. */
  var checked = document.querySelectorAll('input[name="sel"]:checked');
  if (checked.length < 2) return;
  var a = checked[0].value;
  var b = checked[1].value;
  location = "/pair?a=" + encodeURIComponent(a) + "&b=" + encodeURIComponent(b);
}
"""


def render_gallery(runs_dir: Path) -> str:
    """The self-contained all-runs gallery: one row per reportable run
    (headline numbers + a pick checkbox), one muted row per skipped run (the
    reason in place of the numbers — a broken run stays visible), and a
    browser-side compare button. Zero reportable runs: the page says
    `betterbench run`."""
    reportable, skipped = list_runs(runs_dir)
    note = ""
    if not reportable:
        note = ('<p class="empty">No reportable runs here — start one with '
                '<code>betterbench run</code>.</p>')
    rows = []
    for e in reportable:
        try:
            rows.append(_reportable_row(e))
        except Exception as exc:
            rows.append(_degraded_row(e, exc))
    rows += [_skipped_row(e) for e in skipped]
    table = ""
    if rows:
        table = ('\n  <div class="pickbar"><button onclick="compareSel()">'
                 'Compare selected</button>'
                 '<span class="hint">tick two runs, newest first = A</span></div>\n'
                 '  <div class="runs">\n    <table>\n'
                 '      <thead><tr><th>run</th><th>model</th><th>endpoint</th>'
                 '<th>when</th><th>chips</th><th>phases</th>'
                 '<th>combined decode t/s</th><th>ttft p50 (ms)</th>'
                 '<th>aggregate top-conc t/s</th><th></th></tr></thead>\n'
                 '      <tbody>\n' + "\n".join("        " + r for r in rows)
                 + '\n      </tbody>\n    </table>\n  </div>')
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        '<title>BetterBench — all runs</title>\n<style>\n'
        + _PAGE_CSS + '\n</style>\n</head>\n<body>\n<div class="wrap">\n'
        '  <header>\n    <div class="eyebrow">BetterBench — runs</div>\n'
        '    <h1>All runs</h1>\n  </header>\n'
        + note + "\n" + table + '\n</div>\n'
        '<script>\n' + _SEL_JS + "</script>\n</body>\n</html>\n")


# --------------------------------------------------------------------------- #
# The pair page
# --------------------------------------------------------------------------- #
def _muted(text: str) -> str:
    return f'<p class="muted">{_esc(text)}</p>'


def _delta(a, b) -> str:
    """``b − a`` as ``{:+.1f}``, a dash when either side is unmeasured."""
    if a is None or b is None:
        return "—"
    return f"{b - a:+.1f}"


def _decode_series(res: dict, cat: str) -> list:
    """A category's paired `decode_tps` series: the ok passes with a value."""
    recs = (res.get("single_stream") or {}).get(cat) or []
    return [r["decode_tps"] for r in recs
            if r.get("ok") and r.get("decode_tps") is not None]


def _decode_block(a: dict, b: dict) -> str:
    """Decode by category: a paired-t on `decode_tps`, pairing by
    pass-index and truncating to the shorter series, at the default 95%
    confidence — results do not record a confidence, so 95% is the
    documented default. A category missing (or <2 paired values) on a side
    renders ``not measured on <side>`` in that side's cell, no stats; when
    no category has pairs on both sides the block is only the muted note."""
    cats = sorted(set(a.get("single_stream") or {})
                 | set(b.get("single_stream") or {}))
    stats: dict[str, tuple] = {}
    for cat in cats:
        a_s = _decode_series(a, cat)
        b_s = _decode_series(b, cat)
        a_med = float(np.median(a_s)) if a_s else None
        b_med = float(np.median(b_s)) if b_s else None
        p = (paired_compare(a_s, b_s, cat)
             if len(a_s) >= 2 and len(b_s) >= 2 else None)
        stats[cat] = (a_s, b_s, a_med, b_med, p)
    if all(s[4] is None for s in stats.values()):
        return ('<h3 class="sect">Decode by category 95% CI</h3>\n  '
                + _muted("no decode category measured on both sides "
                         "(≥2 paired passes each) — nothing to compare"))
    meds = [m for (_, _, a_m, b_m, _) in stats.values()
            for m in (a_m, b_m) if m is not None]
    top = max(meds) if meds else 1.0
    trs: list[str] = []
    for cat in cats:
        a_s, b_s, a_med, b_med, p = stats[cat]
        a_cell = _num(a_med) if a_med is not None else "not measured on A"
        b_cell = _num(b_med) if b_med is not None else "not measured on B"
        if p is None:
            trs.append(
                f"      <tr><td class=\"cat\">{_esc(cat)}</td>"
                f"<td>{_esc(a_cell)}</td><td>{_esc(b_cell)}</td>"
                f"<td>—</td><td>—</td><td>—</td></tr>")
            continue
        wa = 100.0 * a_med / top
        wb = 100.0 * b_med / top
        verdict = "SIG" if p.significant else "noise"
        trs.append(
            f"      <tr><td class=\"cat\">{_esc(cat)}</td>"
            f"<td>{_num(a_med)}</td><td>{_num(b_med)}</td>"
            f"<td>{p.pct_diff:+.2f}%</td>"
            f"<td>[{p.ci_low_pct:+.1f}, {p.ci_high_pct:+.1f}]</td>"
            f"<td>{verdict}</td></tr>"
            f"      <tr class=\"brow\"><td colspan=\"6\">"
            f'<div class="bars">'
            f'<div class="bar" style="width:{wa:.1f}%"></div>'
            f'<div class="bar bar2" style="width:{wb:.1f}%"></div>'
            f"</div></td></tr>")
    return ('<h3 class="sect">Decode by category 95% CI</h3>\n  <table>\n'
            '    <thead><tr><th>category</th><th>med A</th>'
            '<th>med B</th><th>Δ (B−A)</th>'
            '<th>95% CI</th><th>verdict</th></tr></thead>\n'
            '    <tbody>\n' + "\n".join(trs)
            + "\n    </tbody>\n  </table>")


def _latency_side(res: dict) -> dict:
    """This side's latency summary, read off the `single_rows` builders:
    p50s as the median of the per-category p50s, p99s as the max of the
    per-category p99s, 1%-lows as the min of the per-category 1%-lows.
    Latencies are unpaired between runs (no interleaved `ab`), so each
    metric is a per-side median — the block renders them side by side."""
    rows = single_rows(res)

    def collect(key: str) -> list:
        return [r[key] for r in rows if r.get(key) is not None]

    def med(key: str):
        v = collect(key)
        return float(np.median(v)) if v else None

    def extreme(key: str, fn):
        v = collect(key)
        return fn(v) if v else None

    return {
        "ttft_p50": med("ttft_p50"),
        "ttft_p99": extreme("ttft_p99", max),
        "update_p50": med("update_p50"),
        "update_p99": extreme("update_p99", max),
        "tok_per_update": med("tok_per_update"),
        "itl_low1": extreme("itl_low1", min),
        "itl_med": med("itl_med"),
        "itl_high99": extreme("itl_high99", max),
    }


def _latency_block(a: dict, b: dict) -> str:
    """Latency rows: unpaired medians. The pass-level ITL/gap series do not
    pair across runs without an interleaved `ab`, so medians only — the
    footer says so and points at `betterbench ab` for CIs. Shape-dependent
    columns: stream-update stats when both sides are batched, ITL tails
    otherwise (a batched side naturally reports `None` for ITL)."""
    sa = _latency_side(a)
    sb = _latency_side(b)
    batched = (report_is_batched(single_rows(a))
               and report_is_batched(single_rows(b)))
    if batched:
        shape = (("update p50 (ms)", "update_p50"),
                 ("update p99 (ms)", "update_p99"),
                 ("tok/update", "tok_per_update"))
    else:
        shape = (("itl 1% low (tok/s)", "itl_low1"),
                 ("itl median (tok/s)", "itl_med"),
                 ("itl 99% high (tok/s)", "itl_high99"))
    metrics = (("ttft p50 (ms)", "ttft_p50"),
              ("ttft p99 (ms)", "ttft_p99")) + shape
    all_none = all(sa[k] is None and sb[k] is None
                   for _, k in metrics)
    if all_none:
        return ('<h3 class="sect">Latency (medians only)</h3>\n  '
                + _muted("latency not measured on either run"))
    trs = []
    for label, key in metrics:
        trs.append(
            f"      <tr><td class=\"cat\">{_esc(label)}</td>"
            f"<td>{_num(sa[key])}</td><td>{_num(sb[key])}</td>"
            f"<td>{_delta(sa[key], sb[key])}</td></tr>")
    return ('<h3 class="sect">Latency (medians only)</h3>\n  <table>\n'
            '    <thead><tr><th>metric</th><th>A</th><th>B</th>'
            "<th>Δ (B−A)</th></tr></thead>\n"
            '    <tbody>\n' + "\n".join(trs)
            + "\n    </tbody>\n  </table>\n"
            + _muted("latency rows are unpaired medians — pass-level "
                     "series don't pair across runs; for CIs on those, "
                     "run `betterbench ab`"))


def _safe_rows(builder, res: dict) -> list:
    """One side's phase rows from a builder; an entry the builder can't
    decode (an odd dict) degrades *that side* to no rows, so the block
    falls back to its existing muted ``not measured on <side>`` note
    instead of raising the whole pair page."""
    try:
        return builder(res)
    except Exception:
        return []


def _block_fallback(title: str, note: str) -> str:
    return (f'<h3 class="sect">{_esc(title)}</h3>\n  ' + _muted(note))


def _guarded(title: str, note: str, make) -> str:
    """One comparison block; any exception from the block degrades it to
    its muted ``<phase> not measured on either run`` note rather than
    raising the whole pair page."""
    try:
        return make()
    except Exception:
        return _block_fallback(title, note)


def _phase_block(a: dict, b: dict, label: str, section: str,
                 key: str, cols: tuple) -> str:
    """A prefill (per `target_depth`) / concurrency (per `level`) block:
    the per-key medians side by side with the Δ, *restricted to the keys
    present on both sides*; a key missing on a side renders
    ``not measured on <side>`` in that side's cells. Degraded to a muted
    note when the side lacks the phase entirely (or its entries are
    malformed: that side's rows resolve to none)."""
    builder = prefill_rows if section == "prefill" else concurrency_rows
    rows_a = [r for r in _safe_rows(builder, a) if r.get("skipped") is not True]
    rows_b = [r for r in _safe_rows(builder, b) if r.get("skipped") is not True]
    if not rows_a and not rows_b:
        return (f'<h3 class="sect">{_esc(label)}</h3>\n  '
                + _muted(f"{label.lower()} not measured on either run"))
    if not rows_a:
        return (f'<h3 class="sect">{_esc(label)}</h3>\n  '
                + _muted(f"{label.lower()} not measured on A"))
    if not rows_b:
        return (f'<h3 class="sect">{_esc(label)}</h3>\n  '
                + _muted(f"{label.lower()} not measured on B"))
    da = {r[key]: r for r in rows_a if r.get(key) is not None}
    db = {r[key]: r for r in rows_b if r.get(key) is not None}
    common = sorted(set(da) & set(db))
    head = [f"<th>{'depth' if section == 'prefill' else 'level'}</th>"]
    for name, _ in cols:
        head += [f"<th>A {name}</th>", f"<th>B {name}</th>"]
    trs = []
    for k in common:
        cells = [f"<td class=\"cat\">{_esc(k)}</td>"]
        for _name, fkey in cols:
            va = da[k].get(fkey)
            vb = db[k].get(fkey)
            cells.append(f"<td>{_num(va)}</td>")
            cells.append(f"<td>{_num(vb)}</td>")
            cells.append(f"<td>{_delta(va, vb)}</td>")
        trs.append("      <tr>" + "".join(cells) + "</tr>")
    return (f'<h3 class="sect">{_esc(label)}</h3>\n  <table>\n'
            "    <thead><tr>" + "".join(head) + "</tr></thead>\n"
            "    <tbody>\n" + "\n".join(trs)
            + "\n    </tbody>\n  </table>")


def _combined_block(a: dict, b: dict) -> str:
    """One row: A and B through `combined_score` with *each side's own
    weights* (stated in the caption), Δ guarded for `a` being None/0.
    Unified across the categories each side measured, on the side's own
    `config.weights` — a missing side degrades to a muted note."""
    ca = combined_score(a)
    cb = combined_score(b)
    av = ca["decode"] if ca else None
    bv = cb["decode"] if cb else None
    if av is None and bv is None:
        return ('<h3 class="sect">Combined decode</h3>\n  '
                + _muted("combined decode not measured on either run"))
    a_cell = _num(av) if av is not None else "not measured on A"
    b_cell = _num(bv) if bv is not None else "not measured on B"
    if av in (None, 0) or bv is None:
        d = "—"
    else:
        d = f"{100.0 * (bv - av) / av:+.2f}%"
    return ('<h3 class="sect">Combined decode</h3>\n  <table>\n'
            '    <thead><tr><th>weighted t/s</th><th>A</th><th>B</th>'
            "<th>Δ (B−A)</th></tr></thead>\n"
            '    <tbody>\n'
            f'      <tr><td class="cat">combined</td>'
            f"<td>{_esc(a_cell)}</td><td>{_esc(b_cell)}</td>"
            f"<td>{d}</td></tr>\n"
            '    </tbody>\n  </table>\n'
            + _muted("combined decode — weights are each run's own config"))


def compare_band(a: dict, b: dict) -> str:
    """The pair page's comparison band: decode (paired-t, 95% CI,
    pass-index pairing truncated to the shorter series), latency
    (unpaired medians only), prefill / concurrency (medians per
    depth/level, common keys), and combined decode (each side's own
    weights). Each block degrades to a muted ``phase not measured on
    <side>`` note when that side lacks the phase — never a bare 0."""
    parts = [
        _guarded("Decode by category 95% CI", "decode not measured on either run",
                 lambda: _decode_block(a, b)),
        _guarded("Latency (medians only)",
                 "latency not measured on either run",
                 lambda: _latency_block(a, b)),
        _guarded("Prefill", "prefill not measured on either run",
                 lambda: _phase_block(a, b, "Prefill", "prefill", "target_depth",
                                      (("prompt-tok med", "prompt_tokens_med"),
                                       ("pp med", "pp_med")))),
        _guarded("Concurrency", "concurrency not measured on either run",
                 lambda: _phase_block(a, b, "Concurrency", "concurrency", "level",
                                      (("aggregate t/s", "aggregate_tps"),
                                       ("decode med", "decode_med"),
                                       ("ttft p50 (ms)", "ttft_p50")))),
        _guarded("Combined decode",
                 "combined decode not measured on either run",
                 lambda: _combined_block(a, b)),
    ]
    return '<section class="tiles-cmp">\n  ' + "\n  ".join(parts) \
        + "\n</section>"


_PAIR_CSS = """
  .cmt { display:flex; align-items:center; gap:8px; flex-wrap:wrap;
         font-family:var(--mono); font-size:13px; }
  .cmt select { font-family:var(--mono); font-size:13px; padding:6px 8px;
                border:1px solid var(--grid); border-radius:7px;
                background:var(--page); color:var(--ink); }
  .cmt span { color:var(--muted); }
  .cmt a { font-family:var(--mono); font-size:13px; }
  .warn { border:1px solid var(--s2); border-radius:10px;
          padding:10px 14px; font-family:var(--mono); font-size:12.5px;
          color:var(--ink-2); }
  .chips { max-width:none; }
  .tiles-cmp { background:var(--surface); border:1px solid var(--grid);
              border-radius:10px; padding:12px 16px 4px; }
  .tiles-cmp h3.sect { font-family:var(--sans); font-size:14px;
                      margin:18px 0 6px; color:var(--ink); }
  .tiles-cmp .muted { color:var(--muted); font-family:var(--mono);
                     font-size:12px; margin:4px 0 12px; }
  .tiles-cmp tbody td { color:var(--ink-2); white-space:normal; }
  .tiles-cmp td.cat { color:var(--ink); }
  tr.brow td { border-top:0; padding:2px 12px 8px 0; }
  .bars { display:flex; flex-direction:column; gap:3px; }
  .bar { height:6px; background:var(--s1); border-radius:3px;
        min-width:2px; }
  .bar.bar2 { background:var(--s2); }
  h2 { font-size:18px; margin:4px 0 6px; }
  iframe.runpane { width:100%; border:1px solid var(--grid);
                  border-radius:10px; min-height:70vh; }
"""

_PAIR_JS = """/* The in-situ A/B toolbar: changing either select navigates to
   GET /pair?a=<selA value>&b=<selB value> via a plain relative-path
   assignment — it works on any port; no URL construction needed.
   Both selects carry every other reportable run's URL-encoded slug;
   the swap is a plain anchor to the swapped pair, so it needs no JS. */
var selA = document.getElementById('sel-a');
var selB = document.getElementById('sel-b');
function pairNav() {
  /* Keep A first, B second, exactly as the page header reads them. */
  location = "/pair?a=" + selA.value + "&b=" + selB.value;
}
selA.addEventListener('change', pairNav);
selB.addEventListener('change', pairNav);
/* (handled by the plain `swap` anchor above: no JS state to sync)
   A missing selection can't happen; both selects always default to A
   and B respectively and are never cleared.) */
"""


def _select_options(runs: list[RunEntry], exclude: str) -> str:
    """The `<option>`s for one of the two `select`s: every reportable slug
    except `exclude`, values URL-encoded (`&` → `%26`) and HTML-escaped
    (`&` → `&amp;`) so a slug with an ampersand survives the attribute
    and the later `location` assignment."""
    opts = []
    for e in runs:
        if e.slug == exclude:
            continue
        val = _esc(urllib.parse.quote(str(e.slug), safe=""))
        opts.append(f'<option value="{val}">{_esc(e.slug)}</option>')
    return "\n      ".join(opts)


def render_pair_page(runs_dir: Path, a_slug: str, b_slug: str) -> str:
    """The self-contained pair page. Top to bottom: an A/B toolbar (two
    `select`s — each listing every other reportable run — plus a swap
    link; changing either navigates to `GET /pair?a=…&b=…`), the
    always-on unpaired-in-time banner, the `mismatch_chips` row (hidden
    when empty), `compare_band`, and the two full reports embedded as
    `iframe`s pointing at `/run/<slug>` (the iframes sidestep every
    id/CSS collision between two reports in one document and keep each
    report exactly the standalone shape — the `/run/<slug>` page itself
    is built by the serving layer's string injection, not here)."""
    reportable, _skipped = list_runs(runs_dir)
    by_slug = {e.slug: e for e in reportable}
    missing = [s for s in (a_slug, b_slug) if s not in by_slug]
    if missing:
        raise ValueError(
            f"run {missing[0]!r} has no parseable results.json")
    ops_a = _select_options(reportable, exclude=b_slug)
    ops_b = _select_options(reportable, exclude=a_slug)
    qa = urllib.parse.quote(a_slug, safe="")
    qb = urllib.parse.quote(b_slug, safe="")
    try:
        chips = mismatch_chips(by_slug[a_slug].results,
                               by_slug[b_slug].results)
    except Exception:
        chips = []
    chips_html = ""
    if chips:
        inner = "".join(f'<span class="chip">{_esc(c)}</span>'
                        for c in chips)
        chips_html = f"  <div class=\"chips\">{inner}</div>\n"
    b_res = by_slug[b_slug].results
    try:
        band = compare_band(by_slug[a_slug].results, b_res)
    except Exception:
        band = _muted("the comparison band is unavailable for this pair — "
                      "entire block degraded")
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>BetterBench — {_esc(a_slug)} vs {_esc(b_slug)}</title>\n"
        "<style>\n" + _PAGE_CSS + _PAIR_CSS + "\n</style>\n</head>\n"
        "<body>\n<div class=\"wrap\">\n"
        "  <header>\n"
        "    <div class=\"eyebrow\">BetterBench — compare</div>\n"
        f"    <h1>{_esc(a_slug)} vs {_esc(b_slug)}</h1>\n"
        "  </header>\n"
        "  <div class=\"cmt\">\n"
        f"    <select id=\"sel-a\">\n      {ops_a}\n    </select>\n"
        "    <span>vs</span>\n"
        f"    <select id=\"sel-b\">\n      {ops_b}\n    </select>\n"
        f"    <a href=\"/pair?a={qb}&b={qa}\">swap ⇄</a>\n"
        "  </div>\n"
        "  <div class=\"warn\">cross-file compare is unpaired in time; "
        "for a drift-cancelled verdict run `betterbench ab`. "
        "Paired decode stats below use pass-index pairing "
        "truncated to the shorter series.</div>\n"
        + chips_html
        + f"  {band}\n"
        f"  <h2>Run A — {a_slug}</h2>\n"
        f'  <iframe class="runpane" src="/run/{qa}" '
        "title=\"Report A\"></iframe>\n"
        f"  <h2>Run B — {b_slug}</h2>\n"
        f'  <iframe class="runpane" src="/run/{qb}" '
        "title=\"Report B\"></iframe>\n"
        "</div>\n"
        "<script>\n" + _PAIR_JS + "\n</script>\n</body>\n</html>\n")
