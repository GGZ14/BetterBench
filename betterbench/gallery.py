"""List the runs, summarize each one, and render the all-runs gallery page.

A *reportable* run is a subdirectory of the runs dir whose `results.json`
parses. Its headline numbers are recomputed through the same row builders
(:mod:`betterbench.report`) the markdown and HTML reports use, so the gallery
can never drift from them. Everything here is a pure function over a
directory — the serving layer (run page / pair page) sits on top.
"""
from __future__ import annotations

import html
import json
import math
import statistics
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from .html_report import _pretty_ts
from .report import (combined_score, concurrency_rows, phases_present,
                     single_rows)


@dataclass(frozen=True)
class RunEntry:
    """One subdirectory of the runs directory, classified."""
    slug: str
    path: Path
    results: dict | None      # None when results.json is missing or corrupt
    error: str | None         # why, when not reportable


def list_runs(runs_dir: Path) -> tuple[list[RunEntry], list[RunEntry]]:
    """``(reportable, skipped)`` over the immediate subdirectories of
    ``runs_dir``, slug-descending (timestamp-prefixed names: newest first).

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
    res = e.results
    env = res["env"]
    cfg = res.get("config", {}) or {}
    rows = single_rows(res)
    comb = combined_score(res, rows)
    ttfts = [r["ttft_p50"] for r in rows if r["ttft_p50"] is not None]
    conc = concurrency_rows(res)
    chips = [str(env.get("model", "?")),
             str(env.get("endpoint", "?")),
             "corpus v" + str(res.get("corpus_version", "?"))]
    chips.append("greedy" if cfg.get("greedy")
                 else f"temp {cfg.get('temperature')}")
    chips.append("cold prefix cache (nonce)" if cfg.get("unique_nonce")
                 else "warm prefix cache")
    for k, v in (env.get("notes") or {}).items():
        chips.append(f"{k}: {v}")
    gl = gpu_label(env)
    if gl:
        chips.append(gl)
    ctx = env.get("max_model_len")
    if ctx:
        chips.append(f"{int(ctx):,} tok context")
    return {
        "name": e.slug,
        "model": env["model"],
        "endpoint": env["endpoint"],
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


def _reportable_row(e: RunEntry) -> str:
    m = run_manifest(e)
    chips = "".join(f'<span class="chip">{_esc(c)}</span>' for c in m["chips"])
    val = urllib.parse.quote(str(e.slug), safe="")
    return (f'<tr><td class="slug"><a href="/run/{_esc(e.slug)}">'
            f'{_esc(e.slug)}</a></td>'
            f'<td>{_esc(m["model"])}</td>'
            f'<td class="ep">{_esc(m["endpoint"])}</td>'
            f'<td class="when">{_esc(m["timestamp"])}</td>'
            f'<td class="chips">{chips}</td>'
            f'<td>{_esc(", ".join(m["phases"]) or "—")}</td>'
            f'<td>{_fmt(m["combined_decode"])}</td>'
            f'<td>{_fmt(m["ttft_p50"])}</td>'
            f'<td>{_fmt(m["aggregate_top_conc"])}</td>'
            f'<td class="pick"><input type="checkbox" name="sel" '
            f'value="{_esc(val)}"></td></tr>')


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
    rows = [_reportable_row(e) for e in reportable]
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
