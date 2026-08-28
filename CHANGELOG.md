# Changelog

## 0.3.0

The report now says only what the measurement supports.

### The headline: ITL was wrong on every speculative-decoding server

Through 0.2.3, inter-token latency was derived by multiplying every SSE update gap by one
per-run scalar, `n_chunks / completion_tokens`. Under speculative decoding (MTP, EAGLE,
Medusa, n-gram, DFlash2) several accepted tokens land in one update — they arrived together,
in the same network write, so there is no time *between* them. Applying an average correction
to a chunk-structured distribution keeps the median and destroys both tails.

BetterBench now reports the measured update gap and the tokens per update, and emits no
per-token latency when there is none to emit.

### What moved, and what did not

| metric | changed? |
|---|---|
| `decode t/s`, `total t/s`, `PP t/s`, `TTFT` | **No.** From the server's own `usage` counts over wall clock; they never touched update gaps. Directly comparable to every 0.2.x figure. |
| ITL columns, **one token per update** | Numerically within ~0.2%. 0.2.3 applied a residual sub-threshold rescale even to near-1:1 streams; 0.3.0 reports the raw measured gap. Categories with exactly 1.000 tokens/update are unchanged. |
| ITL columns, **several tokens per update** | **Gone**, replaced by `update p50 / p99 (ms)` + `tok/update`. Every published ITL figure from a speculative rig was an artifact — inflated at the fast end, suppressed at the slow end by the tokens-per-update factor (2.1–7.1× in one archive). |

Old `results.json` files still render: the 0.2.3 scale was a single per-run constant and both
counts were recorded, so the measured gaps are recovered exactly as
`itl_ms × completion_tokens ÷ n_chunks`. No re-running required. `RESULTS_SCHEMA` is now 2.

### Also in this release

- **Sample-size gate wired up.** `enough_samples_for_percentile` was imported and never
  called while three documents claimed under-sampled percentiles were flagged. They are now
  marked `†`, with the shortfall recorded under `sample_gate`. Expect daggers: TTFT p99 fails
  the rule at every pass count the tool offers, and thin categories fail it on the new tail
  metric too — batching by 4× means 4× fewer gap samples.
- **A/B latency row fixed.** It negated the inputs, which made the divisor negative: the `Δ%`
  printed with the opposite sign convention from the decode row directly above it, and the CI
  bounds came out back to front. It also pairs on the measured update gap now, so it works at
  all on a speculative server. The hardcoded `1.0% MDE` and `95% CI` in the A/B report use
  the configured values.
- **Reasoning / answer split and TTFA.** TTFT on a thinking model is time to the first
  *thinking* token. TTFA is the wait a reader feels. A run cut off before `</think>` reports
  `unknown`, never `reasoning_tokens=0, answer_tokens=everything`.
- **`--note KEY=VALUE`.** Image, quant, KV dtype, TP size now travel inside `results.json`.
  `ab` gets an environment block too — it had none.
- **A pytest suite**, and mock-server knobs for tokens-per-update, injected stalls, missing
  usage, reasoning channels and truncation. The batched path was previously untestable.
