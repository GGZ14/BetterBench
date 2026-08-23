"""Turn a results.json into a standalone, self-contained HTML report.

No template engine and no network assets: the output is one file that opens
offline and carries its own CSS, JS and data. Charts are hand-built SVG; the
tables are rendered server-side so the report still reads with JS disabled.

The numbers come from the same row builders the markdown report uses
(:mod:`betterbench.report`), so the two can never drift apart.
"""
from __future__ import annotations

import html
import json
import math
from datetime import datetime

from . import __version__
from .report import (combined_score, concurrency_rows, prefill_rows,
                     single_rows)


def _num(x):
    """JSON-safe float: NaN/inf/None all become None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _esc(x) -> str:
    return html.escape(str(x), quote=True)


def _fmt(x, d=1, unit="", dash="—"):
    v = _num(x)
    return dash if v is None else f"{v:,.{d}f}{unit}"


def _chip(text, swatch=None):
    dot = f'<span class="swatch" style="background:{swatch}"></span>' if swatch else ""
    return f'<span class="chip">{dot}{text}</span>'


def _pretty_ts(raw: str) -> str:
    try:
        return datetime.fromisoformat(raw).strftime("%d %b %Y · %H:%M")
    except (ValueError, TypeError):
        return _esc(raw or "")


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #
def _header(results, cfg, env) -> str:
    sampling = "greedy" if cfg.get("greedy") else f"temp {cfg.get('temperature')}"
    cache = "cold prefix cache (nonce)" if cfg.get("unique_nonce") else "warm prefix cache"
    chips = [
        _chip(_esc(env.get("model", "?")), "var(--s1)"),
        _chip(f'<code>{_esc(env.get("endpoint", "?"))}</code>'),
        _chip(f'corpus v{_esc(results.get("corpus_version", "?"))}'),
        _chip(f'{_esc(cfg.get("runs_per_category"))} passes/cat'),
        _chip(_esc(sampling)),
        _chip(_esc(cache)),
    ]
    gpu = env.get("gpu") or {}
    if gpu:
        label = gpu.get("nvidia_smi") or gpu.get("rocm_smi_productname") or gpu.get("vendor")
        if label:
            chips.append(_chip(_esc(label)))
    ctx = env.get("max_model_len")
    if ctx:
        chips.append(_chip(f"{int(ctx):,} tok context"))
    return f"""  <header>
    <div class="eyebrow">BetterBench {_esc(__version__)} · {_pretty_ts(env.get("timestamp"))} · {_esc(env.get("host", "?"))}</div>
    <h1>{_esc(env.get("model", "Benchmark run"))}</h1>
    <div class="runbar">{"".join(chips)}</div>
  </header>"""


def _tile(label, value, unit, sub):
    u = f" <small>{_esc(unit)}</small>" if unit else ""
    return f"""      <div class="tile">
        <span class="label">{_esc(label)}</span>
        <span class="val">{value}{u}</span>
        <span class="sub">{sub}</span>
      </div>"""


def _tiles(comb, conc, pre, cfg) -> str:
    t = []
    if comb:
        t.append(_tile("Combined decode", _fmt(comb["decode"]), "t/s",
                       "weighted across categories"))
        t.append(_tile("Combined ITL 1% low", _fmt(comb["itl_low1"]), "t/s",
                       "the trustworthy tail metric"))
        t.append(_tile("Combined TTFT p50", _fmt(comb["ttft_p50"], 0), "ms",
                       "single-stream, batch = 1"))
    if conc:
        top = conc[-1]
        t.append(_tile(f"Aggregate @ {top['level']} concurrent",
                       _fmt(top["aggregate_tps"]), "t/s",
                       f"{top['ok']}/{top['requests']} ok"))
    live = [p for p in pre if not p["skipped"]]
    if live:
        deep = live[-1]
        t.append(_tile(f"Prefill @ {int(deep['target_depth']):,} tok",
                       _fmt(deep["pp_med"], 0), "t/s", "median prompt processing"))
    if not t:
        return ""
    return f'  <section class="tiles">\n{chr(10).join(t)}\n  </section>'


def _figure(fid, title, desc, legend=(), note="") -> str:
    leg = ""
    if legend:
        items = "".join(
            f'<span><span class="swatch" style="background:{c}"></span>{_esc(n)}</span>'
            for n, c in legend)
        leg = f'<div class="legend">{items}</div>'
    nt = f'<p class="note">{note}</p>' if note else ""
    return f"""  <figure>
    <figcaption>
      <span class="t">{_esc(title)}</span>
      <span class="d">{desc}</span>
    </figcaption>
    {leg}
    <div class="chartbox" id="{fid}"></div>
    {nt}
  </figure>"""


def _table(caption, headers, body_rows) -> str:
    head = "".join(f"<th>{_esc(h)}</th>" for h in headers)
    body = "\n".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>"
                     for r in body_rows)
    return f"""      <table>
        <caption>{_esc(caption)}</caption>
        <thead><tr>{head}</tr></thead>
        <tbody>
{body}
        </tbody>
      </table>"""


def _tables(rows, conc, pre, env) -> str:
    out = []
    if rows:
        out.append(_table(
            "Single-stream, batch = 1",
            ["category", "passes", "TTFT p50", "TTFT p99", "PP t/s med",
             "ITL 1% low", "ITL med", "ITL 99% high", "decode med", "±IQR", "CV"],
            [[_esc(r["category"].replace("_", " ")), r["runs"], _fmt(r["ttft_p50"]),
              _fmt(r["ttft_p99"]), _fmt(r["pp_med"]), _fmt(r["itl_low1"]),
              _fmt(r["itl_med"]), _fmt(r["itl_high99"]), _fmt(r["decode_med"]),
              _fmt(r["decode_iqr"]), _fmt((r["decode_cv"] or 0) * 100, 1, "%")]
             for r in rows]))
    if conc:
        out.append(_table(
            "Concurrency sweep",
            ["level", "ok/req", "aggregate t/s", "TTFT p50", "TTFT p99", "per-req decode med"],
            [[c["level"], f"{c['ok']}/{c['requests']}", _fmt(c["aggregate_tps"]),
              _fmt(c["ttft_p50"]), _fmt(c["ttft_p99"]), _fmt(c["decode_med"])]
             for c in conc]))
    if pre:
        body = []
        for d in pre:
            if d["skipped"]:
                body.append([f"{int(d['target_depth']):,}", "—", "—", "—", "<em>skipped</em>", "—"])
            else:
                body.append([f"{int(d['target_depth']):,}", _fmt(d["prompt_tokens_med"], 0),
                             _fmt(d["ttft_p50"]), _fmt(d["pp_low1"]), _fmt(d["pp_med"]),
                             _fmt(d["pp_p99"])])
        out.append(_table(
            "Prefill sweep · cold prefix cache",
            ["target depth", "prompt tok med", "TTFT p50", "PP 1% low", "PP median", "PP 99% high"],
            body))
    if not out:
        return ""
    return f"""  <details open>
    <summary>Full numbers</summary>
    <div class="tscroll">
{chr(10).join(out)}
    </div>
  </details>"""


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def render_html(results: dict) -> str:
    cfg = results.get("config", {}) or {}
    env = results.get("env", {}) or {}
    rows = single_rows(results)
    conc = concurrency_rows(results)
    pre = prefill_rows(results)
    comb = combined_score(results, rows)

    live_pre = [p for p in pre if not p["skipped"]]
    skipped = [int(p["target_depth"]) for p in pre if p["skipped"]]

    data = {
        "cats": [r["category"].replace("_", " ") for r in rows],
        "runs": [r["runs"] for r in rows],
        "decode": [_num(r["decode_med"]) for r in rows],
        "decodeIqr": [_num(r["decode_iqr"]) for r in rows],
        "cv": [_num((r["decode_cv"] or 0) * 100) for r in rows],
        "itlLow": [_num(r["itl_low1"]) for r in rows],
        "itlMed": [_num(r["itl_med"]) for r in rows],
        "itlHigh": [_num(r["itl_high99"]) for r in rows],
        "ttft50": [_num(r["ttft_p50"]) for r in rows],
        "combined": _num(comb["decode"]) if comb else None,
        "conc": {
            "levels": [c["level"] for c in conc],
            "agg": [_num(c["aggregate_tps"]) for c in conc],
            "t50": [_num(c["ttft_p50"]) for c in conc],
            "t99": [_num(c["ttft_p99"]) for c in conc],
            "per": [_num(c["decode_med"]) for c in conc],
            "ok": [c["ok"] for c in conc],
            "req": [c["requests"] for c in conc],
        },
        "pre": {
            "depths": [int(p["target_depth"]) for p in live_pre],
            "tok": [_num(p["prompt_tokens_med"]) for p in live_pre],
            "low": [_num(p["pp_low1"]) for p in live_pre],
            "med": [_num(p["pp_med"]) for p in live_pre],
            "high": [_num(p["pp_p99"]) for p in live_pre],
            "t50": [_num(p["ttft_p50"]) for p in live_pre],
        },
    }

    figs = []
    if rows:
        passes_n = rows[0]["runs"] if rows else 0
        figs.append(_figure(
            "cb-decode", "Decode throughput by category",
            f"Median per-pass decode t/s at batch = 1, {passes_n} passes per category. "
            "The dashed line is the weighted combined score.",
            note="Hover a bar for its IQR and coefficient of variation — a high CV means "
                 "the category's passes disagree, so read small differences there with care."))
        figs.append(_figure(
            "cb-itl", "Inter-token latency range by category",
            "Each bar spans the 1% low to the 99% high instantaneous token rate, with a "
            "tick at the median. Wide bars stutter; narrow bars feel smooth.",
            legend=[("1% low → 99% high", "var(--s1-mid)"), ("median", "var(--ink)")]))
    if conc and len(data["conc"]["levels"]) > 1:
        figs.append(_figure(
            "cb-conc", "Aggregate throughput under concurrency",
            "Total tokens/sec across all in-flight requests as load rises — where this "
            "flattens is the throughput knee."))
        figs.append(_figure(
            "cb-ttft", "Time-to-first-token under concurrency",
            "Queueing shows up here first: p50 is the typical wait, p99 the tail.",
            legend=[("TTFT p50", "var(--s1)"), ("TTFT p99", "var(--s2)")],
            note="With a modest request count per level, p99 rests on very few "
                 "observations — treat a spike as a prompt to re-run, not a conclusion."))
    if len(live_pre) > 1:
        note = ""
        if skipped:
            note = ("Skipped depth" + ("s " if len(skipped) > 1 else " ")
                    + ", ".join(f"{d:,}" for d in skipped)
                    + " — deeper than the model's context window.")
        figs.append(_figure(
            "cb-prefill", "Prompt processing throughput by depth",
            "Median prefill t/s (prompt tokens ÷ TTFT) at increasing input depth, cold "
            "prefix cache. The band spans the 1% low to the 99% high.",
            note=note))

    page = _TEMPLATE
    page = page.replace("__TITLE__", _esc(f'BetterBench — {env.get("model", "run")}'))
    page = page.replace("__HEADER__", _header(results, cfg, env))
    page = page.replace("__TILES__", _tiles(comb, conc, pre, cfg))
    page = page.replace("__FIGURES__", "\n".join(figs))
    page = page.replace("__TABLES__", _tables(rows, conc, pre, env))
    page = page.replace("__FOOTER__", _footer(results, env, cfg))
    page = page.replace("__DATA__", json.dumps(data))
    return page


def _footer(results, env, cfg) -> str:
    weights = cfg.get("weights", {}) or {}
    w = ", ".join(f"{k} {v}" for k, v in weights.items())
    ch = env.get("corpus_hash")
    bits = [f"Generated by BetterBench {_esc(__version__)} from a corpus "
            f"v{_esc(results.get('corpus_version', '?'))} run."]
    if ch:
        bits.append(f"Corpus hash <code>{_esc(ch)}</code>.")
    if w:
        bits.append(f"Combined-score weights — {_esc(w)}.")
    bits.append("Results are only comparable within a corpus version. ITL (token-rich) "
                "is the trustworthy tail metric; per-run p99 needs the run count to "
                "support it — see METHODOLOGY.md §sample-size.")
    return "  <footer>" + " ".join(bits) + "</footer>"


# --------------------------------------------------------------------------- #
# The page shell: tokens, layout, and a small hand-rolled SVG chart core.
# Placeholders (__TITLE__ etc.) are filled by render_html.
# --------------------------------------------------------------------------- #
_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    color-scheme: light;
    --page:#f9f9f7; --surface:#fcfcfb;
    --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
    --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10);
    --s1:#2a78d6; --s2:#eb6834;
    --s1-mid:#86b6ef; --s1-soft:rgba(42,120,214,0.13);
    --tip-bg:#ffffff;
    --sans:system-ui,-apple-system,"Segoe UI",sans-serif;
    --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
  }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) {
      color-scheme: dark;
      --page:#0d0d0d; --surface:#1a1a19;
      --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
      --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
      --s1:#3987e5; --s2:#d95926;
      --s1-mid:#256abf; --s1-soft:rgba(57,135,229,0.16);
      --tip-bg:#232322;
    }
  }
  :root[data-theme="dark"] {
    color-scheme: dark;
    --page:#0d0d0d; --surface:#1a1a19;
    --ink:#ffffff; --ink-2:#c3c2b7; --muted:#898781;
    --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10);
    --s1:#3987e5; --s2:#d95926;
    --s1-mid:#256abf; --s1-soft:rgba(57,135,229,0.16);
    --tip-bg:#232322;
  }

  * { box-sizing:border-box; }
  body { background:var(--page); color:var(--ink); font-family:var(--sans);
         line-height:1.55; margin:0; padding:32px 20px 72px; }
  .wrap { max-width:1080px; margin:0 auto; display:flex; flex-direction:column; gap:26px; }

  header { display:flex; flex-direction:column; gap:13px; }
  .eyebrow { font-family:var(--mono); font-size:11px; letter-spacing:.13em;
             text-transform:uppercase; color:var(--muted); }
  h1 { font-size:clamp(26px,4vw,36px); line-height:1.12; margin:0;
       letter-spacing:-.02em; text-wrap:balance; }
  .runbar { display:flex; flex-wrap:wrap; gap:8px; }
  .chip { font-family:var(--mono); font-size:12px; color:var(--ink-2);
          border:1px solid var(--border); border-radius:6px; padding:4px 9px;
          background:var(--surface); display:inline-flex; align-items:center; gap:7px; }
  .chip code { font-family:var(--mono); font-size:12px; }
  .swatch { width:9px; height:9px; border-radius:2px; flex:none; }

  .tiles { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:13px; }
  .tile { background:var(--surface); border:1px solid var(--border); border-radius:10px;
          padding:15px 16px; display:flex; flex-direction:column; gap:4px; }
  .tile .label { font-family:var(--mono); font-size:10.5px; letter-spacing:.1em;
                 text-transform:uppercase; color:var(--muted); }
  .tile .val { font-size:26px; font-weight:600; letter-spacing:-.02em; line-height:1.15; }
  .tile .val small { font-size:14px; font-weight:400; color:var(--ink-2); letter-spacing:0; }
  .tile .sub { font-family:var(--mono); font-size:11.5px; color:var(--muted); }

  figure { margin:0; background:var(--surface); border:1px solid var(--border);
           border-radius:10px; padding:18px 18px 14px;
           display:flex; flex-direction:column; gap:2px; }
  figcaption { display:flex; flex-direction:column; gap:3px; margin-bottom:8px; }
  figcaption .t { font-size:15.5px; font-weight:600; letter-spacing:-.01em; }
  figcaption .d { font-size:13px; color:var(--ink-2); max-width:76ch; }
  .legend { display:flex; gap:16px; flex-wrap:wrap; margin:2px 0 10px; }
  .legend span { font-family:var(--mono); font-size:12px; color:var(--ink-2);
                 display:inline-flex; align-items:center; gap:6px; }
  .chartbox { position:relative; width:100%; }
  .chartbox svg { display:block; width:100%; height:auto; overflow:visible; }
  .note { font-size:12.5px; color:var(--muted); margin:9px 0 0; }

  text { font-family:var(--mono); }
  .tick { font-size:10.5px; fill:var(--muted); }
  .axtitle { font-size:10.5px; fill:var(--muted); letter-spacing:.08em; }
  .dlabel { font-size:11px; font-weight:600; }
  .gridline { stroke:var(--grid); stroke-width:1; }
  .baseline { stroke:var(--axis); stroke-width:1; }
  .refline { stroke:var(--muted); stroke-width:1; stroke-dasharray:4 4; }
  .cross { stroke:var(--axis); stroke-width:1; stroke-dasharray:3 3; }

  .tip { position:absolute; pointer-events:none; opacity:0; transition:opacity .1s ease;
         z-index:5; background:var(--tip-bg); border:1px solid var(--border);
         border-radius:7px; padding:8px 10px; box-shadow:0 3px 14px rgba(0,0,0,.13);
         font-family:var(--mono); font-size:11.5px; color:var(--ink);
         white-space:nowrap; min-width:132px; }
  .tip .th { color:var(--muted); font-size:10px; letter-spacing:.08em;
             text-transform:uppercase; margin-bottom:5px; }
  .tip .row { display:flex; align-items:center; gap:10px; justify-content:space-between; }
  .tip .nm { display:inline-flex; align-items:center; gap:6px; color:var(--ink-2); }
  .tip b { font-weight:600; }

  details { background:var(--surface); border:1px solid var(--border);
            border-radius:10px; padding:0 18px; }
  summary { cursor:pointer; padding:15px 0; font-size:14.5px; font-weight:600;
            list-style:none; display:flex; align-items:center; gap:9px; }
  summary::-webkit-details-marker { display:none; }
  summary::before { content:"\25B8"; color:var(--muted); font-family:var(--mono); }
  details[open] summary::before { content:"\25BE"; }
  summary:focus-visible { outline:2px solid var(--s1); outline-offset:3px; border-radius:4px; }
  .tscroll { overflow-x:auto; padding-bottom:14px; }
  table { border-collapse:collapse; width:100%; font-size:12.5px; font-family:var(--mono);
          font-variant-numeric:tabular-nums; margin-bottom:10px; }
  caption { text-align:left; font-family:var(--sans); font-size:13px; color:var(--ink-2);
            padding:10px 0 8px; }
  th, td { padding:6px 11px 6px 0; text-align:right; white-space:nowrap; }
  th:first-child, td:first-child { text-align:left; }
  thead th { color:var(--muted); font-weight:500; font-size:10.5px; letter-spacing:.07em;
             text-transform:uppercase; border-bottom:1px solid var(--axis); }
  tbody tr + tr td { border-top:1px solid var(--grid); }
  tbody td:first-child { color:var(--ink-2); }

  footer { font-size:12.5px; color:var(--muted); border-top:1px solid var(--border);
           padding-top:16px; }
  footer code { font-family:var(--mono); font-size:12px; color:var(--ink-2); }
  noscript p { font-size:13px; color:var(--ink-2); }
  @media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
</style>
</head>
<body>
<div class="wrap">
__HEADER__
__TILES__
<noscript><p>Charts need JavaScript — the full numbers are in the tables below.</p></noscript>
__FIGURES__
__TABLES__
__FOOTER__
</div>
<script>
"use strict";
const D = __DATA__;
const NS = "http://www.w3.org/2000/svg";
const el = (n, a) => { const e = document.createElementNS(NS, n);
  for (const k in a) e.setAttribute(k, a[k]); return e; };
const fmt = (v, d) => (v === null || v === undefined) ? "—"
  : v.toLocaleString("en-US", { minimumFractionDigits: d, maximumFractionDigits: d });
const has = v => v !== null && v !== undefined;
const maxOf = arrs => Math.max(...arrs.flat().filter(has));
const niceTop = v => { if (!(v > 0)) return 1;
  const p = Math.pow(10, Math.floor(Math.log10(v)) - 1);
  return Math.ceil(v / (2 * p)) * 2 * p; };
const box = id => document.getElementById(id);

function mkTip(b) { const t = document.createElement("div"); t.className = "tip";
  b.appendChild(t); return t; }
function showTip(tip, b, x, y, html) {
  tip.innerHTML = html; tip.style.opacity = "1";
  let left = x + 14;
  if (left + tip.offsetWidth > b.clientWidth) left = x - tip.offsetWidth - 14;
  tip.style.left = Math.max(0, left) + "px";
  tip.style.top = Math.max(0, y - tip.offsetHeight - 10) + "px";
}
const row = (color, name, val) =>
  `<div class="row"><span class="nm">${color
    ? `<span class="swatch" style="background:${color}"></span>` : ""}${name}</span><b>${val}</b></div>`;

function frame(svg, M, iw, ih, yMin, yMax, yTitle, ticks) {
  for (let k = 0; k <= (ticks || 4); k++) {
    const v = yMin + ((yMax - yMin) * k) / (ticks || 4);
    const y = M.t + ih - (ih * (v - yMin)) / (yMax - yMin);
    svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y, class: "gridline" }));
    const tx = el("text", { x: M.l - 10, y: y + 3.5, class: "tick", "text-anchor": "end" });
    tx.textContent = fmt(v, v < 10 ? 1 : 0); svg.appendChild(tx);
  }
  svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: M.t + ih, y2: M.t + ih, class: "baseline" }));
  const ay = el("text", { x: M.l - 10, y: M.t - 5, class: "axtitle", "text-anchor": "end" });
  ay.textContent = yTitle; svg.appendChild(ay);
}
function xLabels(svg, labels, X, y) {
  labels.forEach((lab, i) => {
    const t = el("text", { x: X(i), y: y, class: "tick", "text-anchor": "middle" });
    t.textContent = lab; svg.appendChild(t);
  });
}

/* --- grouped / single bars, zero-based --- */
function bars(id, o) {
  const b = box(id); if (!b) return;
  const W = 900, H = o.height || 340, M = { t: 20, r: 16, b: 50, l: 64 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
  const n = o.x.length, yMax = o.yMax;
  const Y = v => M.t + ih - (ih * v) / yMax;
  frame(svg, M, iw, ih, 0, yMax, o.yTitle);
  const band = iw / n, ns = o.series.length, GAP = 2;
  const bw = Math.min(40, (band - 24 - GAP * (ns - 1)) / ns);
  xLabels(svg, o.x, i => M.l + band * i + band / 2, M.t + ih + 20);
  if (has(o.ref)) {
    const y = Y(o.ref);
    svg.appendChild(el("line", { x1: M.l, x2: M.l + iw, y1: y, y2: y, class: "refline" }));
    const t = el("text", { x: M.l + iw, y: y - 6, class: "tick", "text-anchor": "end" });
    t.textContent = o.refLabel; svg.appendChild(t);
  }
  const tip = mkTip(b);
  o.x.forEach((lab, i) => {
    const cx = M.l + band * i + band / 2;
    const start = cx - (ns * bw + (ns - 1) * GAP) / 2;
    o.series.forEach((s, si) => {
      const v = s.v[i]; if (!has(v)) return;
      const x = start + si * (bw + GAP);
      const r = el("rect", { x: x, y: Y(v), width: bw, height: Math.max(2, ih - (Y(v) - M.t)),
                             rx: 4, ry: 4, fill: s.color });
      r.style.cursor = "crosshair";
      const html = `<div class="th">${lab}</div>` +
        o.series.map(q => row(ns > 1 ? q.color : null, q.name, o.tipVal(q.v[i]))).join("") +
        (o.extra ? o.extra(i) : "");
      r.addEventListener("pointerenter", () => {
        const rc = svg.getBoundingClientRect();
        showTip(tip, b, ((x + bw / 2) / W) * rc.width, (Y(v) / H) * rc.height, html);
      });
      r.addEventListener("pointerleave", () => { tip.style.opacity = "0"; });
      svg.appendChild(r);
    });
  });
  b.appendChild(svg);
}

/* --- floating range bars with a median tick --- */
function rangeBars(id, o) {
  const b = box(id); if (!b) return;
  const W = 900, H = 340, M = { t: 20, r: 16, b: 50, l: 64 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
  const n = o.x.length, yMax = o.yMax;
  const Y = v => M.t + ih - (ih * v) / yMax;
  frame(svg, M, iw, ih, 0, yMax, o.yTitle);
  const band = iw / n, bw = Math.min(34, band - 26);
  xLabels(svg, o.x, i => M.l + band * i + band / 2, M.t + ih + 20);
  const tip = mkTip(b);
  o.x.forEach((lab, i) => {
    const lo = o.low[i], hi = o.high[i], md = o.med[i];
    if (!has(lo) || !has(hi)) return;
    const cx = M.l + band * i + band / 2, x = cx - bw / 2;
    const r = el("rect", { x: x, y: Y(hi), width: bw, height: Math.max(3, Y(lo) - Y(hi)),
                           rx: 4, ry: 4, fill: "var(--s1-mid)" });
    r.style.cursor = "crosshair";
    svg.appendChild(r);
    if (has(md)) svg.appendChild(el("line", { x1: x, x2: x + bw, y1: Y(md), y2: Y(md),
                                              stroke: "var(--ink)", "stroke-width": 2 }));
    const html = `<div class="th">${lab}</div>` +
      row(null, "99% high", o.tipVal(hi)) + row(null, "median", o.tipVal(md)) +
      row(null, "1% low", o.tipVal(lo)) + (o.extra ? o.extra(i) : "");
    r.addEventListener("pointerenter", () => {
      const rc = svg.getBoundingClientRect();
      showTip(tip, b, (cx / W) * rc.width, (Y(hi) / H) * rc.height, html);
    });
    r.addEventListener("pointerleave", () => { tip.style.opacity = "0"; });
  });
  b.appendChild(svg);
}

/* --- line chart with optional band --- */
function line(id, o) {
  const b = box(id); if (!b) return;
  const W = 900, H = 320, M = { t: 20, r: 24, b: 52, l: 64 };
  const iw = W - M.l - M.r, ih = H - M.t - M.b;
  const svg = el("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": o.aria });
  const n = o.x.length, yMin = o.yMin, yMax = o.yMax;
  const X = i => M.l + (n === 1 ? iw / 2 : (iw * i) / (n - 1));
  const Y = v => M.t + ih - (ih * (v - yMin)) / (yMax - yMin);
  frame(svg, M, iw, ih, yMin, yMax, o.yTitle);
  xLabels(svg, o.x, X, M.t + ih + 20);
  if (o.xTitle) {
    const t = el("text", { x: M.l, y: H - 8, class: "axtitle" });
    t.textContent = o.xTitle; svg.appendChild(t);
  }
  if (o.band && o.band.low.every(has) && o.band.high.every(has)) {
    const up = o.band.high.map((v, i) => (i ? "L" : "M") + X(i) + " " + Y(v)).join(" ");
    const dn = o.band.low.map((v, i) => "L" + X(n - 1 - i) + " " + Y(o.band.low[n - 1 - i])).join(" ");
    svg.appendChild(el("path", { d: up + " " + dn + " Z", fill: "var(--s1-soft)", stroke: "none" }));
  }
  svg.appendChild(el("path", {
    d: o.v.map((v, i) => (i ? "L" : "M") + X(i) + " " + Y(v)).join(" "),
    fill: "none", stroke: "var(--s1)", "stroke-width": 2,
    "stroke-linejoin": "round", "stroke-linecap": "round" }));
  o.v.forEach((v, i) => svg.appendChild(el("circle", {
    cx: X(i), cy: Y(v), r: 5, fill: "var(--s1)", stroke: "var(--surface)", "stroke-width": 2 })));
  const cross = el("line", { class: "cross", y1: M.t, y2: M.t + ih, opacity: 0 });
  svg.appendChild(cross);
  const hit = el("rect", { x: M.l, y: M.t, width: iw, height: ih, fill: "transparent" });
  svg.appendChild(hit);
  b.appendChild(svg);
  const tip = mkTip(b);
  hit.addEventListener("pointermove", ev => {
    const rc = svg.getBoundingClientRect();
    const px = ((ev.clientX - rc.left) / rc.width) * W;
    let i = 0, best = Infinity;
    for (let k = 0; k < n; k++) { const d = Math.abs(X(k) - px); if (d < best) { best = d; i = k; } }
    cross.setAttribute("x1", X(i)); cross.setAttribute("x2", X(i)); cross.setAttribute("opacity", 1);
    showTip(tip, b, (X(i) / W) * rc.width, (Y(o.v[i]) / H) * rc.height,
      `<div class="th">${o.tipHead(i)}</div>` + o.tipBody(i));
  });
  hit.addEventListener("pointerleave", () => { tip.style.opacity = "0"; cross.setAttribute("opacity", 0); });
}

/* ------------------------------ build ------------------------------ */
if (D.cats.length) {
  bars("cb-decode", {
    aria: "Median decode throughput per category.",
    x: D.cats, yTitle: "DECODE T/S (MEDIAN)",
    yMax: niceTop(maxOf([D.decode])), height: 350,
    ref: D.combined, refLabel: "combined " + fmt(D.combined, 1),
    series: [{ name: "decode t/s", color: "var(--s1)", v: D.decode }],
    tipVal: v => fmt(v, 1) + " t/s",
    extra: i => row(null, "±IQR", fmt(D.decodeIqr[i], 1)) +
                row(null, "CV", fmt(D.cv[i], 1) + "%") +
                row(null, "passes", D.runs[i])
  });
  rangeBars("cb-itl", {
    aria: "Inter-token latency range per category, 1% low to 99% high.",
    x: D.cats, yTitle: "INSTANTANEOUS T/S",
    yMax: niceTop(maxOf([D.itlHigh])),
    low: D.itlLow, med: D.itlMed, high: D.itlHigh,
    tipVal: v => fmt(v, 1) + " t/s",
    extra: i => row(null, "spread", has(D.itlHigh[i]) && has(D.itlLow[i])
      ? fmt(D.itlHigh[i] - D.itlLow[i], 1) + " t/s" : "—")
  });
}
if (D.conc.levels.length > 1) {
  line("cb-conc", {
    aria: "Aggregate throughput as concurrency rises.",
    x: D.conc.levels.map(String), xTitle: "CONCURRENCY LEVEL",
    yTitle: "AGGREGATE T/S", yMin: 0, yMax: niceTop(maxOf([D.conc.agg])),
    v: D.conc.agg,
    tipHead: i => "concurrency " + D.conc.levels[i] + " · " + D.conc.ok[i] + "/" + D.conc.req[i] + " ok",
    tipBody: i => row(null, "aggregate", fmt(D.conc.agg[i], 1) + " t/s") +
                  row(null, "per-req", fmt(D.conc.per[i], 1) + " t/s") +
                  row(null, "TTFT p50", fmt(D.conc.t50[i], 1) + " ms")
  });
  bars("cb-ttft", {
    aria: "TTFT p50 and p99 by concurrency level.",
    x: D.conc.levels.map(String), yTitle: "TTFT (MS)",
    yMax: niceTop(maxOf([D.conc.t99, D.conc.t50])), height: 300,
    series: [{ name: "p50", color: "var(--s1)", v: D.conc.t50 },
             { name: "p99", color: "var(--s2)", v: D.conc.t99 }],
    tipVal: v => fmt(v, 1) + " ms",
    extra: i => row(null, "requests", D.conc.ok[i] + "/" + D.conc.req[i])
  });
}
if (D.pre.depths.length > 1) {
  const lo = Math.min(...D.pre.low.filter(has).concat(D.pre.med.filter(has)));
  const hi = Math.max(...D.pre.high.filter(has).concat(D.pre.med.filter(has)));
  const pad = (hi - lo) * 0.25 || hi * 0.1;
  line("cb-prefill", {
    aria: "Prefill throughput by input depth.",
    x: D.pre.depths.map(d => d >= 1000 ? (d / 1000) + "k" : String(d)),
    xTitle: "TARGET INPUT DEPTH (TOKENS)", yTitle: "PP T/S (MEDIAN)",
    yMin: Math.max(0, lo - pad), yMax: hi + pad,
    v: D.pre.med, band: { low: D.pre.low, high: D.pre.high },
    tipHead: i => D.pre.depths[i].toLocaleString() + " target · "
                  + fmt(D.pre.tok[i], 0) + " actual tok",
    tipBody: i => row(null, "99% high", fmt(D.pre.high[i], 1)) +
                  row(null, "median", fmt(D.pre.med[i], 1)) +
                  row(null, "1% low", fmt(D.pre.low[i], 1)) +
                  row(null, "TTFT p50", fmt(D.pre.t50[i], 0) + " ms")
  });
}
</script>
</body>
</html>
"""
