"""Orchestration: single-stream, concurrency sweep, and interleaved paired A/B.

Uses the stdlib streaming client wrapped in threads (asyncio.to_thread) for
real parallelism under the concurrency sweep.
"""
from __future__ import annotations

import asyncio
import random
import time

import numpy as np

from .client import RunResult, is_context_length_error, stream_chat
from .config import Config
from .corpus import Prompt, nonce, with_nonce
from .metrics import (paired_ci_halfwidth_pct, paired_compare,
                      required_pairs_for_mde)
from .prefill import make_prefill_messages


async def _one(endpoint: str, model: str, p: Prompt, cfg: Config,
               rng: random.Random) -> RunResult:
    msgs = with_nonce(p.messages, nonce(rng)) if cfg.unique_nonce else p.messages
    return await stream_chat(
        endpoint, model, msgs,
        max_tokens=p.max_tokens, temperature=cfg.effective_temp(),
        top_p=cfg.top_p, top_k=cfg.top_k, seed=cfg.seed,
        category=p.category, prompt_id=p.id, timeout=cfg.timeout_s)


# --------------------------------------------------------------------------- #
# Single stream (batch = 1)
# --------------------------------------------------------------------------- #
async def single_stream(endpoint: str, model: str,
                        corpus: dict[str, list[Prompt]], cfg: Config,
                        log=print) -> dict[str, list[dict]]:
    results: dict[str, list[dict]] = {}
    rng = random.Random(1234)
    for cat, prompts in corpus.items():
        log(f"[single] {cat}: warmup {cfg.warmup} + {cfg.runs_per_category} runs")
        for i in range(cfg.warmup):
            await _one(endpoint, model, prompts[i % len(prompts)], cfg, rng)
        recs = []
        for i in range(cfg.runs_per_category):
            p = prompts[i % len(prompts)]
            r = await _one(endpoint, model, p, cfg, rng)
            if not r.ok:
                log(f"  ! {cat}/{p.id}: {r.error}")
            recs.append(r.as_dict())
        results[cat] = recs
    return results


# --------------------------------------------------------------------------- #
# Concurrency sweep
# --------------------------------------------------------------------------- #
async def concurrency_sweep(endpoint: str, model: str,
                            corpus: dict[str, list[Prompt]], cfg: Config,
                            log=print) -> list[dict]:
    flat = [p for ps in corpus.values() for p in ps]
    rng = random.Random(99)
    out = []
    for level in cfg.concurrency_levels:
        log(f"[concurrency] level {level}: {cfg.concurrency_requests} requests")
        sem = asyncio.Semaphore(level)
        recs: list[RunResult] = []
        t0 = time.perf_counter()

        async def worker(idx: int):
            async with sem:
                recs.append(await _one(endpoint, model, flat[idx % len(flat)], cfg, rng))

        await asyncio.gather(*(worker(i) for i in range(cfg.concurrency_requests)))
        wall = time.perf_counter() - t0
        ok = [r for r in recs if r.ok]
        comp = sum((r.completion_tokens or 0) for r in ok)
        out.append({
            "level": level, "requests": len(recs), "ok": len(ok), "wall_s": wall,
            "aggregate_tps": (comp / wall) if wall else 0.0,
            "ttft_ms": [r.ttft_ms for r in ok if r.ttft_ms],
            "decode_tps": [r.decode_tps for r in ok if r.decode_tps],
            # Per-request gap series would dwarf the rest of the file at this
            # request count; the ratio is what the report needs to know whether
            # the level was streaming one token per update.
            "tokens_per_update": [r.tokens_per_update for r in ok
                                  if r.tokens_per_update],
            "batched_runs": sum(1 for r in ok if r.chunk_token_mismatch),
        })
    return out


# --------------------------------------------------------------------------- #
# Prompt-processing (prefill) sweep — throughput vs input depth
# --------------------------------------------------------------------------- #
async def prefill_sweep(endpoint: str, model: str, cfg: Config, log=print,
                        max_ctx: int | None = None) -> list[dict]:
    """Sweep prefill throughput vs input depth.

    Depths that can't fit the model's context window are skipped rather than
    run, so a limited-context server doesn't return HTTP 400s mid-sweep:
      * if `max_ctx` is known, a depth needing more than it (plus the tiny
        decode and a small margin) is skipped up front;
      * regardless, if the server rejects a depth at runtime with a
        context-length error, that depth is marked skipped and abandoned.
    """
    rng = random.Random(2024)
    out = []

    def skip_row(depth: int, reason: str) -> dict:
        return {"target_depth": depth, "skipped": True, "reason": reason,
                "prompt_tokens": [], "ttft_ms": [], "pp_tps": [],
                "tokens_per_update": [], "batched_runs": 0}

    for depth in cfg.prefill_depths:
        need = depth + cfg.prefill_max_tokens + cfg.prefill_ctx_margin
        if max_ctx is not None and need > max_ctx:
            log(f"[prefill] skip depth ~{depth}: needs ~{need} tok > model context {max_ctx}")
            out.append(skip_row(depth, f"exceeds max context {max_ctx} (needs ~{need} tok)"))
            continue

        log(f"[prefill] depth ~{depth} tok: warmup {cfg.prefill_warmup} + {cfg.prefill_runs}")

        async def one():
            msgs = make_prefill_messages(depth, nonce(rng))   # unique nonce => no cache
            return await stream_chat(endpoint, model, msgs,
                                     max_tokens=cfg.prefill_max_tokens, temperature=0.0,
                                     top_p=cfg.top_p, top_k=cfg.top_k, seed=cfg.seed,
                                     category="prefill", prompt_id=f"pp{depth}",
                                     timeout=cfg.timeout_s)

        rejected = False
        recs = []
        for _ in range(cfg.prefill_warmup + cfg.prefill_runs):
            r = await one()
            if is_context_length_error(r.error):
                rejected = True
                break
            recs.append(r)
        if rejected:
            log(f"[prefill] skip depth ~{depth}: server rejected (context length "
                f"exceeded){'' if max_ctx else '; consider --max-model-len'}")
            out.append(skip_row(depth, "server rejected: exceeds context length"))
            continue

        measured = recs[cfg.prefill_warmup:]          # drop warmup
        for r in measured:
            if not r.ok:
                log(f"  ! prefill depth {depth}: {r.error}")
        ok = [r for r in measured if r.ok]
        out.append({
            "target_depth": depth,
            "prompt_tokens": [r.prompt_tokens for r in ok if r.prompt_tokens],
            "ttft_ms": [r.ttft_ms for r in ok if r.ttft_ms],
            "pp_tps": [r.pp_tps for r in ok if r.pp_tps],
            "tokens_per_update": [r.tokens_per_update for r in ok
                                  if r.tokens_per_update],
            "batched_runs": sum(1 for r in ok if r.chunk_token_mismatch),
        })
    return out


# --------------------------------------------------------------------------- #
# Interleaved paired A/B (repeatability path, plan §7)
# --------------------------------------------------------------------------- #
async def paired_ab(endpoint_a: str, endpoint_b: str, model: str,
                    corpus: dict[str, list[Prompt]], cfg: Config,
                    log=print) -> dict:
    flat = [p for ps in corpus.values() for p in ps]
    rng = random.Random(7)
    a_tps: list[float] = []; b_tps: list[float] = []
    a_itl: list[float] = []; b_itl: list[float] = []
    batched = False

    def med(samples):
        return float(np.median(samples)) if samples else None

    async def call(ep, p, msgs):
        return await stream_chat(ep, model, msgs, max_tokens=p.max_tokens,
                                 temperature=cfg.effective_temp(), top_p=cfg.top_p,
                                 top_k=cfg.top_k, seed=cfg.seed, timeout=cfg.timeout_s)

    for i in range(cfg.warmup):
        p = flat[i % len(flat)]
        await call(endpoint_a, p, p.messages); await call(endpoint_b, p, p.messages)

    for i in range(cfg.ab_max_pairs):
        p = flat[i % len(flat)]
        msgs = with_nonce(p.messages, nonce(rng)) if cfg.unique_nonce else p.messages
        first_is_a = (i % 2 == 0)             # counterbalance order
        if first_is_a:
            ra = await call(endpoint_a, p, msgs); rb = await call(endpoint_b, p, msgs)
        else:
            rb = await call(endpoint_b, p, msgs); ra = await call(endpoint_a, p, msgs)
        # Median *stream-update* gap, not per-token ITL: on a speculative server
        # itl_ms is empty by construction, and pairing on it would drop every
        # pair (and used to raise TypeError on the None it left behind).
        ma, mb = med(ra.update_gaps_ms), med(rb.update_gaps_ms)
        if ra.ok and rb.ok and ra.decode_tps and rb.decode_tps \
                and ma is not None and mb is not None:
            a_tps.append(ra.decode_tps); b_tps.append(rb.decode_tps)
            a_itl.append(ma); b_itl.append(mb)
            if ra.chunking == "batched" or rb.chunking == "batched":
                batched = True
        if len(a_tps) >= cfg.ab_min_pairs:
            hw = paired_ci_halfwidth_pct(a_tps, b_tps, cfg.conf)
            if hw <= cfg.target_mde_pct:
                log(f"[ab] CI half-width {hw:.2f}% <= {cfg.target_mde_pct}% "
                    f"after {len(a_tps)} pairs — stopping")
                break

    tps = paired_compare(a_tps, b_tps, "decode_tps", cfg.conf, higher_is_better=True)
    gap_metric = "update_gap_median_ms" if batched else "itl_median_ms"
    itl = paired_compare(a_itl, b_itl, gap_metric, cfg.conf,
                         higher_is_better=False)
    paired_cv = (float(np.std(np.array(b_tps) - np.array(a_tps), ddof=1) / np.mean(a_tps))
                 if len(a_tps) > 1 else 0.0)
    return {
        "endpoint_a": endpoint_a, "endpoint_b": endpoint_b, "model": model,
        "pairs": len(a_tps), "decode_tps": tps.as_dict(),
        "gap_median": {**itl.as_dict(), "batched": batched},
        "conf": cfg.conf, "target_mde_pct": cfg.target_mde_pct,
        "paired_cv": paired_cv,
        "pairs_needed_for_mde": required_pairs_for_mde(paired_cv, cfg.target_mde_pct, cfg.conf),
        "raw": {"a_tps": a_tps, "b_tps": b_tps, "a_itl": a_itl, "b_itl": b_itl},
    }
