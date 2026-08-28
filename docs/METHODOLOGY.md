# BetterBench methodology

## Metrics (three layers)

All timing is captured client-side from the streamed response and cross-checked against the
server's reported `usage`.

- **TTFT** — time from request send to the first content-bearing chunk (ms). Captures prefill
  and, under load, queueing.
- **Stream-update gap** — the wall-clock time between one SSE update and the next (ms).
  This is the measured quantity; everything below is derived from it or from the server's
  own `usage` counts. Reported as **update p50** (the typical rhythm) and **update p99**
  (the stutter).
- **ITL** — per-token inter-token latency (ms), reported **only when the server streams one
  token per update**, in which case the update gap *is* the inter-token latency and is
  emitted unchanged. Shown as tokens/sec so the tails read naturally: **median** = `1/p50`
  and **99% high** = `1/p1`; the **1% low** column is the gaming-style *mean of the worst 1%*
  of instantaneous rates rather than `1/p99` — two different estimators, deliberately, and
  the 0.1% low is computed but not printed. See "Speculative decoding" below for why ITL is
  absent on some servers.
- **Per-run tokens/sec** — `completion_tokens / decode_wall` (decode) and `/ total_wall`
  (includes TTFT). Reported as **median ± IQR**; p1/p99 only when the run count supports it.
- **Prompt-processing (prefill) throughput** — `prompt_tokens / TTFT`. TTFT includes a small
  first-token-decode + network term, negligible for large prompts, so this is the standard
  prefill-throughput approximation. Measured per single-stream category, and swept explicitly
  over input depth (2K → 64K) with a **cold prefix cache** (unique nonce) and a tiny decode
  (`prefill_max_tokens`) so the number reflects prefill, not generation. The target depth is
  approximate (built at ~4 chars/token); the reported depth is the server's actual
  `usage.prompt_tokens`, so throughput is always tied to the real token count. Prefill is
  typically far more repeatable run-to-run than decode. Depths larger than the model's
  context window are **skipped, not run**: the window is auto-detected from `GET /v1/models`
  (`max_model_len`) or set with `--max-model-len`, and any depth needing more than it (plus the
  tiny decode and a small margin) is dropped up front. As a fallback, a context-length HTTP 400
  at runtime also marks that depth skipped rather than aborting the sweep. Skipped depths are
  recorded in `results.json` and shown as `skipped` in the report.

Reasoning-channel tokens (`reasoning` / `reasoning_content`) are counted as generated tokens;
TTFT is time to the first token of *any* channel — which on a thinking model is the first
*thinking* token, a wait nobody experiences. **TTFA** (time to first answer) is reported
alongside it.

### Reasoning / answer split
BetterBench detects the thinking→answering transition two ways: a switch from a
`reasoning_content` / `reasoning` delta field to `content`, which is unambiguous; or a
literal `</think>` inside `content`, found with a carry buffer so a marker straddling two
updates is still caught. TTFA is accurate to update granularity.

The split is apportioned from the server's single `usage.completion_tokens` by character
count, so the token figures are estimates (`_est`) and **under-count answer tokens where
answers are punctuation-dense** — json, code, file_edit — plausibly by 10–20%. The raw
character and update counts are recorded so an exact tokenizer-based split can be computed
later without re-running.

**A run cut off mid-thought is unknown, not an answer.** Qwen3-style templates open `<think>`
in the prompt, so a generation that hits `max_tokens` before emitting `</think>` shows no
marker at all — indistinguishable from a non-thinking model that got truncated. That yields
`reasoning_source="unknown"` with the split and TTFA left as `None`, never zeros: recording
`reasoning_tokens=0, answer_tokens=everything` would credit an answer that never arrived.

### Truncation
This matters more than it sounds. Across one 42-file archive, **84% of runs stop at
`max_tokens`** — 100% of `chat`, 91% of `reasoning`. On a thinking model those runs measure
the thinking phase, not a complete answer. Both reports state the truncation rate; read
per-category numbers with it in mind, and treat cross-model comparison of heavily truncated
categories with suspicion.

### Chunk ↔ token reconciliation
Most servers stream one token per SSE update, but not all. BetterBench records per-update
arrival times and compares the update count to `usage.completion_tokens`, per record.

A stream counts as one-token-per-update when `completion_tokens` exceeds the update count by
no more than two tokens or 2%, whichever is larger. Both bounds matter: `usage` counts tokens
that never carried a delta — the stop token, and on some servers a role-only opening chunk —
so a genuinely 1:1 stream lands one or two tokens above its update count regardless of
length. A ratio alone passes that on a 900-token generation and fails it on a 16-token
prefill probe. The ratio is deliberately tight, because whatever it admits is the worst-case
error in any per-token latency then reported.

Anything else is `chunk_token_mismatch`, and BetterBench reports update gaps and tokens per
update instead. **It never converts an update gap into a per-token latency.** If `usage` is
missing entirely the classification is `unknown` and no per-token latency is reported —
guessing a ratio we cannot see is the same mistake as assuming one.

### Speculative decoding
Speculative decoding (MTP, EAGLE, Medusa, n-gram, DFlash2) verifies several tokens in a
single forward pass and writes all the accepted ones into one update. Those tokens did not
arrive at different times — they arrived together, in the same network write — so for tokens
inside an update, "time between tokens" has no answer, and none is available client-side at
any sampling rate.

Before 0.3.0, BetterBench answered anyway: it multiplied every update gap by one per-run
scalar, `n_chunks / completion_tokens`. That preserves the median and destroys both tails. A
near-zero gap between two updates inside one decode step became a near-infinite per-token
rate; a real stall got divided down — in one archived run a 1296.7 ms stall was reported as
567.3 ms. The error was also directional: the divisor is tokens-per-update, so an arm that
speculates better got a bigger divisor and therefore a lower apparent per-token latency. It
looked better partly for a measurement reason.

Two caveats on the replacement. `update p99` is an *upper bound* on any pause a reader feels,
not a per-token figure, and it is **not comparable across servers that pack different numbers
of tokens into an update** — compare decode t/s for that. And batching by 4x means 4x fewer
gap samples, so the honest tail metric is statistically thinner than the fabricated one it
replaces; thin categories are flagged (see Sample-size honesty).

Decode t/s, total t/s, PP t/s and TTFT are unaffected by any of this. They come from the
server's own `usage` counts over wall clock and never touch update gaps, so they are directly
comparable across versions and across engines.

## Resolving ~1% differences (the point of the tool)

Raw run-to-run noise here is commonly **3–8%** (cache warmth, GPU clock/thermal drift, scheduler
nondeterminism, sampling-induced length variance). To call a ~1% change real, the uncertainty on
the *difference* must be well under 1%. BetterBench gets there by controlling variance and
comparing in pairs:

1. **Paired, interleaved A/B** (`betterbench ab`). A and B run back-to-back on the *same* prompt,
   alternating order, on the same warmed box. The confidence interval is computed on the
   **per-trial difference** (paired-t), so common-mode drift cancels — turning an unresolvable
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
A p1/p99 needs enough samples beyond the tail (`n · tail ≥ ~5`, i.e. 500 observations for a
p99). BetterBench marks every percentile below that threshold with a `†` and records the
shortfall under `sample_gate` in `results.json`, so the claim is checkable after the fact.

The number is marked rather than suppressed. At the default 20 passes, TTFT p99 rests on
n=20 — it has never been supported at any pass count the tool offers, and hiding it would
delete a column people use, including the concurrency-table p99 that detects queueing. Read a
daggered figure as "roughly the worst observed", not as a percentile. The same mark covers
`PP 1% low`, where at 8 prefill runs the "1% low" is literally the single slowest observation
wearing a percentile's label.

## Reproducibility
Each `results.json` embeds: BetterBench + corpus version, sampling params, the endpoint/model,
host/OS, best-effort GPU/driver info, and a content hash of the exact prompts used. Results are
only comparable within a `corpus_version`.
