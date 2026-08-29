"""Turn a results.json into a human-readable markdown report."""
from __future__ import annotations

import numpy as np

from .client import classify_chunking
from .metrics import (Dist, enough_samples_for_percentile, itl_to_rate_samples,
                      summarize)


def run_gaps_ms(r: dict) -> list[float]:
    """Wall-clock gaps between stream updates for one run.

    Schema 2 records them directly. Schema 1 stored `itl_ms`, which was the same
    gaps multiplied by a single per-run scalar `n_chunks / completion_tokens` —
    so the measured series is exactly recoverable, and old results re-render
    under the honest columns instead of being refused.
    """
    gaps = r.get("update_gaps_ms")
    if gaps is not None:
        return gaps
    itl = r.get("itl_ms") or []
    comp, n = r.get("completion_tokens"), r.get("n_chunks")
    if itl and comp and n:
        return [x * comp / n for x in itl]
    return list(itl)


def run_tokens_per_update(r: dict) -> float | None:
    tpu = r.get("tokens_per_update")
    if tpu:
        return tpu
    comp, n = r.get("completion_tokens"), r.get("n_chunks")
    return (comp / n) if comp and n else None


def run_chunking(r: dict) -> str:
    """"per_token" | "batched" | "unknown" for one run.

    Re-derived from the recorded counts rather than trusting the stored
    `chunk_token_mismatch`, so a schema-1 file written under the old 10%
    tolerance is judged by exactly the same rule as a fresh one. The stored flag
    is only a fallback for a record that has no usage counts at all.
    """
    comp, n = r.get("completion_tokens"), r.get("n_chunks")
    if comp and n:
        return classify_chunking(comp, n)[0]
    if r.get("chunk_token_mismatch"):
        return "batched"
    return "unknown"


def run_is_batched(r: dict) -> bool:
    """Did this run pack several tokens into one stream update?"""
    return run_chunking(r) == "batched"


def _collect(recs: list[dict]):
    ok = [r for r in recs if r.get("ok")]
    ttft = [r["ttft_ms"] for r in ok if r.get("ttft_ms")]
    dtps = [r["decode_tps"] for r in ok if r.get("decode_tps")]
    gaps = [x for r in ok for x in run_gaps_ms(r)]
    comp = [r.get("completion_tokens") or 0 for r in ok]
    chunks = [r.get("n_chunks") or 0 for r in ok]
    pp = [r["pp_tps"] for r in ok if r.get("pp_tps")]
    batched = sum(1 for r in ok if run_is_batched(r))
    return ttft, dtps, gaps, comp, chunks, pp, batched


def _fmt(x, unit=""):
    return f"{x:.1f}{unit}" if x is not None else "—"


def _fmt0(x, unit=""):
    return f"{x:.0f}{unit}" if x is not None else "—"


def _fmt2(x, unit=""):
    return f"{x:.2f}{unit}" if x is not None else "—"


def _category_row(cat: str, recs: list[dict]) -> dict:
    ttft, dtps, gaps, comp, chunks, pp, batched = _collect(recs)
    ttft_d = summarize(ttft)
    dtps_d = summarize(dtps)
    gap_d = summarize(gaps)
    pp_d = summarize(pp)
    n_ok = len([r for r in recs if r.get("ok")])
    tot_chunks = int(np.sum(chunks)) if chunks else 0
    row = {
        # Sample counts behind each tail, so the reader (and the gate) can see
        # what a percentile actually rests on.
        "ttft_n": ttft_d.n, "gap_n": gap_d.n,
        "ttft_p99_ok": enough_samples_for_percentile(ttft_d.n, 99),
        "tail_ok": enough_samples_for_percentile(gap_d.n, 99),
        "category": cat,
        "runs": n_ok,
        "tokens": int(np.sum(comp)) if comp else 0,
        "ttft_p50": ttft_d.median, "ttft_p99": ttft_d.p99,
        "pp_med": pp_d.median,
        # Measured, always present: the wall-clock gap between stream updates.
        "update_p50": gap_d.median if gaps else None,
        "update_p99": gap_d.p99 if gaps else None,
        "tok_per_update": (float(np.sum(comp)) / tot_chunks) if tot_chunks else None,
        "batched_runs": batched,
        "batched": batched > 0,
        "decode_med": dtps_d.median, "decode_iqr": dtps_d.iqr, "decode_cv": dtps_d.cv,
    }
    # Per-token ITL exists only when the server streams one token per chunk.
    # When several tokens land in one update they arrived together, so there is
    # no "time between" them to report — and inventing one wrecks both tails.
    if batched or not gaps:
        row.update(itl_low1=None, itl_med=None, itl_high99=None)
    else:
        rate_d = summarize(itl_to_rate_samples(gaps), rate_like=True)
        row.update(itl_low1=rate_d.low_1pct, itl_med=rate_d.median,
                   itl_high99=rate_d.p99)
    return row


# A split is only reported over runs that actually produced one, and only when
# enough of them did. On a thinking model with a tight max_tokens most runs stop
# mid-thought, so "we don't know" is the common answer, not the edge case.
SPLIT_MIN_RUNS = 5
SPLIT_MIN_SHARE = 0.5


def reasoning_rows(results: dict) -> list[dict]:
    """Per-category thinking/answering split. Shared by both reports."""
    rows = []
    for cat, recs in results.get("single_stream", {}).items():
        ok = [r for r in recs if r.get("ok")]
        known = [r for r in ok if r.get("reasoning_source") in ("channel",
                                                                "inline_marker")
                 and r.get("reasoning_tokens_est") is not None]
        answered = [r for r in known if r.get("ttfa_ms") is not None]
        never = sum(1 for r in ok if r.get("truncated_in_reasoning")
                    or r.get("reasoning_source") == "unknown")
        r_tok = sum(r["reasoning_tokens_est"] for r in known)
        tot_tok = r_tok + sum(r.get("answer_tokens_est") or 0 for r in known)
        enough = len(known) >= SPLIT_MIN_RUNS and (
            len(known) / len(ok) >= SPLIT_MIN_SHARE if ok else False)
        rows.append({
            "category": cat,
            "runs": len(ok),
            "known": len(known),
            "answered": len(answered),
            "never_answered": never,
            "reasoning_share": (r_tok / tot_tok) if (enough and tot_tok) else None,
            "ttfa_p50": (summarize([r["ttfa_ms"] for r in answered]).median
                         if len(answered) >= SPLIT_MIN_RUNS else None),
        })
    return rows


def has_reasoning_evidence(rows: list[dict]) -> bool:
    return any(r["known"] for r in rows)


def truncation_summary(results: dict) -> tuple[int, int]:
    """(runs stopped at max_tokens, total ok runs) across the single-stream set."""
    ok = [r for recs in results.get("single_stream", {}).values() for r in recs
          if r.get("ok")]
    return sum(1 for r in ok if r.get("finish_reason") == "length"), len(ok)


def _gate(x, ok, fmt=None) -> str:
    """Render a percentile, daggered when it rests on too few samples."""
    fmt = fmt or _fmt
    txt = fmt(x)
    return txt if (ok or txt == "—") else txt + "†"


GATE_NOTE = (
    "† this percentile rests on fewer samples than `n · tail ≥ 5` requires — a "
    "p99 needs 500 observations, and 20 passes give 20. Read it as \"roughly the "
    "worst observed\", not as a percentile. The full list is under `sample_gate` "
    "in `results.json`.")


def sample_gate(results: dict) -> dict:
    """Which percentiles in this report are under-sampled, and by how much.

    `enough_samples_for_percentile` has existed since 0.1 and was imported by
    this module without ever being called, while three documents claimed
    under-sampled percentiles were flagged. This is the thing that makes the
    claim true, and it is persisted so the claim is checkable after the fact.
    """
    under = []

    def check(section, key, metric, n, pct=99):
        if not enough_samples_for_percentile(n, pct):
            tail = min(pct, 100 - pct) / 100.0
            under.append({"section": section, "key": key, "metric": metric,
                          "n": int(n), "need": int(np.ceil(5 / tail))})

    for r in single_rows(results):
        check("single_stream", r["category"], "ttft_p99", r["ttft_n"])
        check("single_stream", r["category"],
              "update_p99" if r["batched"] else "itl_high99", r["gap_n"])
    for c in concurrency_rows(results):
        check("concurrency", c["level"], "ttft_p99", c["ttft_n"])
    for d in prefill_rows(results):
        if not d["skipped"]:
            check("prefill", d["target_depth"], "pp_p99", d["pp_n"])
    return {"rule": "n * min(pct, 100-pct) / 100 >= 5", "under_sampled": under}


def report_is_batched(rows: list[dict]) -> bool:
    """Does this report need the stream-update columns instead of ITL?"""
    return any(r.get("batched") for r in rows)


def single_rows(results: dict) -> list[dict]:
    """Per-category single-stream rows. Shared by the markdown and HTML reports."""
    return [_category_row(c, r) for c, r in results.get("single_stream", {}).items()]


def concurrency_rows(results: dict) -> list[dict]:
    """Per-level concurrency rows. Shared by the markdown and HTML reports."""
    rows = []
    for s in results.get("concurrency", []):
        td = summarize(s.get("ttft_ms", []))
        dd = summarize(s.get("decode_tps", []))
        rows.append({
            "level": s["level"], "ok": s["ok"], "requests": s["requests"],
            "aggregate_tps": s["aggregate_tps"],
            "ttft_p50": td.median, "ttft_p99": td.p99, "decode_med": dd.median,
            "ttft_n": td.n,
            "ttft_p99_ok": enough_samples_for_percentile(td.n, 99),
        })
    return rows


def prefill_rows(results: dict) -> list[dict]:
    """Per-depth prefill rows. Shared by the markdown and HTML reports."""
    rows = []
    for d in results.get("prefill", []):
        if d.get("skipped"):
            rows.append({"target_depth": d["target_depth"], "skipped": True})
            continue
        pt = summarize(d.get("prompt_tokens", []))
        td = summarize(d.get("ttft_ms", []))
        pp = summarize(d.get("pp_tps", []), rate_like=True)
        rows.append({
            "target_depth": d["target_depth"], "skipped": False,
            "prompt_tokens_med": pt.median, "ttft_p50": td.median,
            "pp_low1": pp.low_1pct, "pp_med": pp.median, "pp_p99": pp.p99,
            "pp_n": pp.n,
            "pp_tail_ok": enough_samples_for_percentile(pp.n, 99),
        })
    return rows


def combined_score(results: dict, rows: list[dict] | None = None) -> dict | None:
    """Weighted combined single-stream score, or None if there are no rows."""
    rows = single_rows(results) if rows is None else rows
    return _combined(rows, results.get("config", {}).get("weights", {}))


PHASE_SECTIONS = (("single_stream", "decode"), ("prefill", "prefill"),
                  ("concurrency", "concurrency"))


def phases_present(results: dict) -> list[str]:
    """Which measured phases this result actually carries.

    Read off the sections themselves, not the config, so a file written before
    phase selection existed reports what is in it.
    """
    return [label for key, label in PHASE_SECTIONS if results.get(key)]


def render_markdown(results: dict) -> str:
    L = []
    fp = results.get("env", {})
    cfg = results.get("config", {})
    phases = phases_present(results)
    L.append(f"# BetterBench report\n")
    L.append(f"- **endpoint**: `{fp.get('endpoint','?')}`  ·  **model**: "
             f"`{fp.get('model','?')}`  ·  **host**: {fp.get('host','?')}")
    bits = [f"- **corpus**: v{results.get('corpus_version','?')}",
            f"**sampling**: {'greedy' if cfg.get('greedy') else 'temp '+str(cfg.get('temperature'))}"]
    # passes/cat describes the single-stream phase; on a prefill- or
    # concurrency-only run it would be a number nothing was measured with.
    if results.get("single_stream"):
        bits.append(f"**passes/cat**: {cfg.get('runs_per_category')}")
    bits.append(f"prefix-cache: {'cold (nonce)' if cfg.get('unique_nonce') else 'warm'}")
    if len(phases) < len(PHASE_SECTIONS):
        bits.append(f"**phases**: {', '.join(phases) or 'none'}")
    L.append("  ·  ".join(bits))
    notes = fp.get("notes") or {}
    if notes:
        L.append("- **notes**: " + "  ·  ".join(f"`{k}={v}`" for k, v in notes.items()))
    gpu = fp.get("gpu", {})
    if gpu:
        L.append(f"- **gpu**: {gpu.get('vendor','?')} {gpu.get('nvidia_smi', gpu.get('rocm_smi_productname',''))}")
    L.append("")

    single = results.get("single_stream", {})
    if single:
        rows = single_rows(results)
        batched = report_is_batched(rows)
        L.append("## Single-stream (batch = 1)\n")
        if batched:
            L.append("This server packs several tokens into one stream update "
                     "(speculative decoding), so there is no per-token latency to "
                     "report — the tokens in an update arrive together. **update "
                     "p50/p99** is the measured wall-clock gap between updates "
                     "(p99 is the stutter); **tok/update** is how many tokens land "
                     "per update. TTFT in ms; decode = per-run tok/s.\n")
            L.append("| category | passes | TTFT p50 | TTFT p99 | PP t/s (med) | update p50 (ms) | update p99 (ms) | tok/update | decode t/s (med) | ±IQR | CV |")
            L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
            for r in rows:
                L.append(f"| {r['category']} | {r['runs']} | {_fmt(r['ttft_p50'])} | "
                         f"{_gate(r['ttft_p99'], r['ttft_p99_ok'])} | {_fmt(r.get('pp_med'))} | "
                         f"{_fmt(r['update_p50'])} | {_gate(r['update_p99'], r['tail_ok'])} | "
                         f"{_fmt2(r['tok_per_update'])} | {_fmt(r['decode_med'])} | "
                         f"{_fmt(r['decode_iqr'])} | {r['decode_cv']*100:.1f}% |")
        else:
            L.append("ITL columns are tokens/sec: **1% low** = slowest tokens (stutter), "
                     "**median**, **99% high** = fastest. This server streams one token "
                     "per update, so the update gap *is* the inter-token latency. "
                     "TTFT in ms; decode = per-run tok/s.\n")
            L.append("| category | passes | TTFT p50 | TTFT p99 | PP t/s (med) | ITL 1% low | ITL median | ITL 99% high | decode t/s (med) | ±IQR | CV |")
            L.append("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|")
            for r in rows:
                L.append(f"| {r['category']} | {r['runs']} | {_fmt(r['ttft_p50'])} | "
                         f"{_gate(r['ttft_p99'], r['ttft_p99_ok'])} | {_fmt(r.get('pp_med'))} | "
                         f"{_gate(r['itl_low1'], r['tail_ok'])} | {_fmt(r['itl_med'])} | "
                         f"{_gate(r['itl_high99'], r['tail_ok'])} | {_fmt(r['decode_med'])} | "
                         f"{_fmt(r['decode_iqr'])} | {r['decode_cv']*100:.1f}% |")
        # combined weighted
        weights = cfg.get("weights", {})
        comb = _combined(rows, weights)
        if comb:
            lead = (f"update p99 ≈ **{_fmt(comb['update_p99'])} ms**" if batched
                    else f"ITL 1%-low ≈ **{_fmt(comb['itl_low1'])} t/s**")
            L.append(f"\n**Combined (weighted {', '.join(f'{k}:{v}' for k,v in weights.items() if k in single)})** "
                     f"— decode t/s median ≈ **{_fmt(comb['decode'])}**, {lead}, "
                     f"TTFT p50 ≈ **{_fmt0(comb['ttft_p50'])} ms**")
        n_batched = sum(r["batched_runs"] for r in rows)
        n_runs = sum(r["runs"] for r in rows)
        if n_batched:
            L.append(f"\n*{n_batched} of {n_runs} runs streamed several tokens per "
                     f"update (`chunk_token_mismatch`). Per-token ITL is not reported "
                     f"for them — see METHODOLOGY.md §chunk-token.*")
        L.append("")

    rrows = reasoning_rows(results)
    if has_reasoning_evidence(rrows):
        L.append("## Reasoning / answer split\n")
        L.append("A per-token rate cannot see how much of a run was spent "
                 "thinking. Two configs with identical decode t/s can take very "
                 "different times to reach an answer. **TTFA** is time to the "
                 "first *answer* token — the wait a reader actually feels.\n")
        L.append("| category | runs w/ split | reasoning share (est) | TTFA p50 (ms) | never reached answer |")
        L.append("|---|--:|--:|--:|--:|")
        for r in rrows:
            share = (f"{r['reasoning_share']*100:.0f}%"
                     if r["reasoning_share"] is not None else "—")
            L.append(f"| {r['category']} | {r['known']}/{r['runs']} | {share} | "
                     f"{_fmt(r['ttfa_p50'])} | {r['never_answered']}/{r['runs']} |")
        L.append(f"\n*A `—` means too few runs reached an answer to say "
                 f"(fewer than {SPLIT_MIN_RUNS}, or under half the passes). Runs "
                 f"cut off before any answer began are counted, not folded in: "
                 f"crediting their output as an answer would flatter the result. "
                 f"Token counts are apportioned by character count, so the share "
                 f"is an estimate — punctuation-dense answers (json, code) are "
                 f"under-counted.*")
        L.append("")

    trunc, tot = truncation_summary(results)
    if tot and trunc:
        L.append(f"*Stopped at `max_tokens`: **{trunc}/{tot}** runs "
                 f"({trunc/tot*100:.0f}%). On a thinking model a truncated run "
                 f"measures the thinking phase, not a complete answer.*\n")

    sweep = results.get("concurrency", [])
    if sweep:
        L.append("## Concurrency sweep\n")
        L.append("| level | ok/req | aggregate t/s | TTFT p50 | TTFT p99 | per-req decode t/s (med) |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        for c in concurrency_rows(results):
            L.append(f"| {c['level']} | {c['ok']}/{c['requests']} | "
                     f"{c['aggregate_tps']:.1f} | {_fmt(c['ttft_p50'])} | "
                     f"{_gate(c['ttft_p99'], c['ttft_p99_ok'])} | {_fmt(c['decode_med'])} |")
        L.append("")

    prefill = results.get("prefill", [])
    if prefill:
        L.append("## Prompt processing (prefill) sweep\n")
        L.append("Prefill throughput = prompt tokens ÷ TTFT, at increasing input depth "
                 "(tiny decode, cold prefix cache). PP t/s columns: 1% low / median / 99% high.\n")
        L.append("| target depth | prompt tokens (med) | TTFT p50 (ms) | PP t/s 1% low | PP t/s median | PP t/s 99% high |")
        L.append("|--:|--:|--:|--:|--:|--:|")
        for d in prefill_rows(results):
            if d["skipped"]:
                L.append(f"| {d['target_depth']} | — | — | — | _skipped_ | — |")
                continue
            pt_med = f"{d['prompt_tokens_med']:.0f}" if d["prompt_tokens_med"] is not None else "—"
            L.append(f"| {d['target_depth']} | {pt_med} | {_fmt(d['ttft_p50'])} | "
                     f"{_gate(d['pp_low1'], d['pp_tail_ok'])} | {_fmt(d['pp_med'])} | "
                     f"{_gate(d['pp_p99'], d['pp_tail_ok'])} |")
        skipped = [str(d["target_depth"]) for d in prefill if d.get("skipped")]
        if skipped:
            ctx = results.get("env", {}).get("max_model_len")
            ctx_txt = f" (model context {ctx} tokens)" if ctx else ""
            L.append(f"\n*Skipped {', '.join(skipped)} — input depth exceeds the model's "
                     f"context window{ctx_txt}; raise it or pass `--max-model-len` to bench "
                     f"deeper.*")
        L.append("")

    L.append("---")
    if sample_gate(results)["under_sampled"]:
        L.append(f"*{GATE_NOTE}*\n")
    L.append("*Generated by BetterBench. See METHODOLOGY.md §sample-size.*")
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
            "update_p50": wavg("update_p50"), "update_p99": wavg("update_p99"),
            "ttft_p50": wavg("ttft_p50")}


def render_ab_markdown(ab: dict) -> str:
    L = ["# BetterBench — paired A/B\n"]
    L.append(f"- **A**: `{ab['endpoint_a']}`  vs  **B**: `{ab['endpoint_b']}`  ·  "
             f"model `{ab['model']}`  ·  **{ab['pairs']}** interleaved pairs")
    mde = ab.get("target_mde_pct", 1.0)
    L.append(f"- paired CV of decode diff: {ab['paired_cv']*100:.2f}%  ·  "
             f"pairs needed for {mde}% MDE: ~{ab['pairs_needed_for_mde']}\n")
    gap = ab.get("gap_median") or ab.get("itl_median") or {}
    gap_label = ("Stream-update gap, median (B vs A)" if gap.get("batched")
                 else "ITL median smoothness (B vs A)")
    for key, label, d in (("decode_tps", "Decode throughput (B vs A)", ab["decode_tps"]),
                          ("gap_median", gap_label, gap)):
        if not d:
            continue
        conf = int(round(d.get("conf", 0.95) * 100))
        L.append(f"### {label}")
        direction = "" if d.get("higher_is_better", True) else "  ·  lower is better"
        L.append(f"- Δ = **{d['pct_diff']:+.2f}%**  ({conf}% CI [{d['ci_low_pct']:+.2f}%, "
                 f"{d['ci_high_pct']:+.2f}%]){direction}")
        L.append(f"- **{d['verdict']}**\n")
    return "\n".join(L)
