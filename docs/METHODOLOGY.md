# BetterBench methodology

## Metrics (three layers)

All timing is captured client-side from the streamed response and cross-checked against the
server's reported `usage`.

- **TTFT** — time from request send to the first content-bearing chunk (ms). Captures prefill
  and, under load, queueing.
- **ITL** — per-token inter-token latency (ms). Reported as tokens/sec so the tails read
  naturally: **1% low** = `1/p99(ITL)` (slowest tokens / stutter), **median** = `1/p50`,
  **99% high** = `1/p1`. We also compute the gaming-style *mean of the worst 1% / 0.1%* of
  instantaneous rates.
- **Per-run tokens/sec** — `completion_tokens / decode_wall` (decode) and `/ total_wall`
  (includes TTFT). Reported as **median ± IQR**; p1/p99 only when the run count supports it.
- **Prompt-processing (prefill) throughput** — `prompt_tokens / TTFT`. TTFT includes a small
  first-token-decode + network term, negligible for large prompts, so this is the standard
  prefill-throughput approximation. Measured per single-stream category, and swept explicitly
  over input depth (2K → 64K) with a **cold prefix cache** (unique nonce) and a tiny decode
  (`prefill_max_tokens`) so the number reflects prefill, not generation. The target depth is
  approximate (built at ~4 chars/token); the reported depth is the server's actual
  `usage.prompt_tokens`, so throughput is always tied to the real token count. Prefill is
  typically far more repeatable run-to-run than decode.

Reasoning-channel tokens (`reasoning` / `reasoning_content`) are counted as generated tokens;
TTFT is time to the first token of *any* channel.

### Chunk ↔ token reconciliation
Most servers stream one token per SSE chunk, but not all. BetterBench records per-chunk arrival
times and, at the end, compares chunk count to `usage.completion_tokens`. If they differ by
>10% it flags `chunk_token_mismatch` and scales per-chunk gaps to per-token so ITL stays
comparable across engines.

## Resolving ~1% differences (the point of the tool)

Raw run-to-run noise here is commonly **3–8%** (cache warmth, GPU clock/thermal drift, scheduler
nondeterminism, sampling-induced length variance). To call a ~1% change real, the uncertainty on
the *difference* must be well under 1%. BetterBench gets there by controlling variance and
comparing in pairs:

1. **Paired, interleaved A/B** (`betterbench ab`). A and B run back-to-back on the *same* prompt,
   alternating order, on the same warmed box. The confidence interval is computed on the
   **per-trial difference** (bootstrap), so common-mode drift cancels — turning an unresolvable
   ±5% absolute into a ±0.3% CI on Δ.
2. **Greedy + fixed seed** (A/B default). Identical output token counts → no length variance.
3. **Nonce prefixes** defeat the prefix cache so prefill timing is honest. A warm-cache mode can
   be reported separately, never mixed.
4. **Power analysis / run-to-confidence.** From the observed coefficient of variation of the
   paired difference, BetterBench estimates the pairs needed for a target minimum-detectable
   effect (default 1% @ 95%) as `n ≈ ((z_α+z_β)·CV/effect)²`, and can stop early once the Δ CI is
   tighter than the MDE.
5. **Significance verdicts.** Every comparison prints `Δ ± CI` and refuses to declare a winner
   when the CI straddles zero.
6. **Null-test gate** (`tools/self_test.py`). Comparing a config against itself must report
   *not significant*. If the tool manufactures a phantom win, its noise model is wrong.

### Practical tips for a clean 1% measurement
- Run on an otherwise-idle box; pin the CPU governor to performance.
- Warm to a **thermal plateau** before measuring; ideally pin GPU clocks (disable boost/DVFS) or
  at least record temp/clocks and discard runs that drift.
- Keep the client on localhost or a dedicated low-latency link.
- Prefer **ITL median** as the lead signal (thousands of token samples ⇒ tight CI); per-run t/s
  needs ~100+ runs for a trustworthy p99.

## Sample-size honesty
A p1/p99 needs enough samples beyond the tail (`n · tail ≥ ~5`). BetterBench flags any percentile
below that threshold and falls back to median + IQR rather than printing a meaningless tail.

## Reproducibility
Each `results.json` embeds: BetterBench + corpus version, sampling params, the endpoint/model,
host/OS, best-effort GPU/driver info, and a content hash of the exact prompts used. Results are
only comparable within a `corpus_version`.
