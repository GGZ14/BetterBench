"""Turn a results.json into a human-readable markdown report."""
from __future__ import annotations

import numpy as np

from .metrics import (Dist, enough_samples_for_percentile, itl_to_rate_samples,
                      summarize)


def _collect(recs: list[dict]):
    ttft = [r["ttft_ms"] for r in recs if r.get("ok") and r.get("ttft_ms")]
    dtps = [r["decode_tps"] for r in recs if r.get("ok") and r.get("decode_tps")]
    itl = [x for r in recs if r.get("ok") for x in (r.get("itl_ms") or [])]
    comp = [r.get("completion_tokens") or 0 for r in recs if r.get("ok")]
    pp = [r["pp_tps"] for r in recs if r.get("ok") and r.get("pp_tps")]
    return ttft, dtps, itl, comp, pp


def _fmt(x, unit=""):
    return f"{x:.1f}{unit}" if x is not None else "—"


def _category_row(cat: str, recs: list[dict]) -> dict:
    ttft, dtps, itl, comp, pp = _collect(recs)
    ttft_d = summarize(ttft)
    dtps_d = summarize(dtps)
    rate = itl_to_rate_samples(itl)               # instantaneous tok/s per token
    rate_d = summarize(rate, rate_like=True)
    pp_d = summarize(pp)
    return {
        "category": cat,
        "runs": len([r for r in recs if r.get("ok")]),
        "tokens": int(np.sum(comp)) if comp else 0,
        "ttft_p50": ttft_d.median, "ttft_p99": ttft_d.p99,
        "pp_med": pp_d.median,
        "itl_low1": rate_d.low_1pct, "itl_med": rate_d.median, "itl_high99": rate_d.p99,
        "decode_med": dtps_d.median, "decode_iqr": dtps_d.iqr, "decode_cv": dtps_d.cv,
    }


def render_markdown(results: dict) -> str:
    L = []
    fp = results.get("env", {})
    cfg = results.get("config", {})
    L.append(f"# BetterBench report\n")
    L.append(f"- **endpoint**: `{fp.get('endpoint','?')}`  ·  **model**: "
             f"`{fp.get('model','?')}`  ·  **host**: {fp.get('host','?')}")
    L.append(f"- **corpus**: v{results.get('corpus_version','?')}  ·  "
             f"**sampling**: {'greedy' if cfg.get('greedy') else 'temp '+str(cfg.get('temperature'))}"
             f"  ·  **runs/cat**: {cfg.get('runs_per_category')}  ·  "
             f"prefix-cache: {'cold (nonce)' if cfg.get('unique_nonce') else 'warm'}")
    gpu = fp.get("gpu", {})
    if gpu:
        L.append(f"- **gpu**: {gpu.get('vendor','?')} {gpu.get('nvidia_smi', gpu.get('rocm_smi_productname',''))}")
    L.append("")

    single = results.get("single_stream", {})
    if single:
        rows = [_category_row(c, r) for c, r in single.items()]
        L.append("## Single-stream (batch = 1)\n")
        L.append("ITL columns are tokens/sec: **1% low** = slowest tokens (stutter), "
                 "**median**, **99% high** = fastest. TTFT in ms; decode = per-run tok/s.\n")
        L.append("| category | runs | TTFT p50 | TTFT p99 | PP t/s (med) | ITL 1% low | ITL median | ITL 99% high | decode t/s (med) | ±IQR | CV |")
        L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
        for r in rows:
            L.append(f"| {r['category']} | {r['runs']} | {_fmt(r['ttft_p50'])} | "
                     f"{_fmt(r['ttft_p99'])} | {_fmt(r.get('pp_med'))} | {_fmt(r['itl_low1'])} | {_fmt(r['itl_med'])} | "
                     f"{_fmt(r['itl_high99'])} | {_fmt(r['decode_med'])} | "
                     f"{_fmt(r['decode_iqr'])} | {r['decode_cv']*100:.1f}% |")
        # combined weighted
        weights = cfg.get("weights", {})
        comb = _combined(rows, weights)
        if comb:
            L.append(f"\n**Combined (weighted {', '.join(f'{k}:{v}' for k,v in weights.items() if k in single)})** "
                     f"— decode t/s median ≈ **{comb['decode']:.1f}**, ITL 1%-low ≈ "
                     f"**{comb['itl_low1']:.1f} t/s**, TTFT p50 ≈ **{comb['ttft_p50']:.0f} ms**")
        L.append("")

    sweep = results.get("concurrency", [])
    if sweep:
        L.append("## Concurrency sweep\n")
        L.append("| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        for s in sweep:
            td = summarize(s.get("ttft_ms", []))
            dd = summarize(s.get("decode_tps", []))
            L.append(f"| {s['level']} | {s['ok']}/{s['requests']} | "
                     f"{s['aggregate_tps']:.1f} | {_fmt(td.median)} | {_fmt(td.p99)} | "
                     f"{_fmt(dd.median)} |")
        L.append("")

    prefill = results.get("prefill", [])
    if prefill:
        L.append("## Prompt processing (prefill) sweep\n")
        L.append("Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth "
                 "(tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.\n")
        L.append("| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        for d in prefill:
            if d.get("skipped"):
                L.append(f"| {d['target_depth']} | — | — | — | _skipped_ | — |")
                continue
            pt = summarize(d.get("prompt_tokens", []))
            td = summarize(d.get("ttft_ms", []))
            pp = summarize(d.get("pp_tps", []), rate_like=True)
            pt_med = f"{pt.median:.0f}" if pt.median is not None else "—"
            L.append(f"| {d['target_depth']} | {pt_med} | {_fmt(td.median)} | "
                     f"{_fmt(pp.low_1pct)} | {_fmt(pp.median)} | {_fmt(pp.p99)} |")
        skipped = [str(d["target_depth"]) for d in prefill if d.get("skipped")]
        if skipped:
            ctx = results.get("env", {}).get("max_model_len")
            ctx_txt = f" (model context {ctx} tokens)" if ctx else ""
            L.append(f"\n*Skipped {', '.join(skipped)} — input depth exceeds the model's "
                     f"context window{ctx_txt}; raise it or pass `--max-model-len` to bench "
                     f"deeper.*")
        L.append("")

    L.append("---\n*Generated by BetterBench. Percentiles below their reliable "
             "sample size are flagged in `results.json`; ITL (token-rich) is the "
             "trustworthy tail metric — see METHODOLOGY.md §sample-size.*")
    return "\n".join(L)


def _combined(rows, weights):
    if not rows:
        return None
    tot = sum(weights.get(r["category"], 0) for r in rows)
    if tot <= 0:
        tot = len(rows)
        w = {r["category"]: 1 for r in rows}
    else:
        w = weights
    def wavg(key):
        vals = [(w.get(r["category"], 0), r[key]) for r in rows if r[key] is not None]
        s = sum(wt for wt, _ in vals)
        return sum(wt * v for wt, v in vals) / s if s else None
    return {"decode": wavg("decode_med"), "itl_low1": wavg("itl_low1"),
            "ttft_p50": wavg("ttft_p50")}


def render_ab_markdown(ab: dict) -> str:
    L = ["# BetterBench — paired A/B\n"]
    L.append(f"- **A**: `{ab['endpoint_a']}`  vs  **B**: `{ab['endpoint_b']}`  ·  "
             f"model `{ab['model']}`  ·  **{ab['pairs']}** interleaved pairs")
    L.append(f"- paired CV of decode diff: {ab['paired_cv']*100:.2f}%  ·  "
             f"pairs needed for {1.0}% MDE: ~{ab['pairs_needed_for_mde']}\n")
    for key, label in (("decode_tps", "Decode throughput (B vs A)"),
                       ("itl_median", "ITL median smoothness (B vs A)")):
        d = ab[key]
        L.append(f"### {label}")
        L.append(f"- Δ = **{d['pct_diff']:+.2f}%**  (95% CI [{d['ci_low_pct']:+.2f}%, "
                 f"{d['ci_high_pct']:+.2f}%])")
        L.append(f"- **{d['verdict']}**\n")
    return "\n".join(L)
