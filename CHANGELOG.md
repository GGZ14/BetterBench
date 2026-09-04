# Changelog

## Unreleased

### `PP t/s` is gone from the single-stream table

The single-stream phase reported a per-category `PP t/s (med)` — prompt tokens ÷ TTFT over the
corpus prompts. On prompts that short, TTFT is dominated by fixed per-request overhead (queue,
tokenize, the first decode step) rather than by prefill work, so the number came out far below
what the server actually does: on the same run, `chat` read 2,021 t/s in the single-stream
table against 4,713 t/s at the 2K depth of the prefill sweep. It was a wrong number in a
table of right ones, so it has been dropped from both the markdown and HTML reports.

The **prefill sweep is unchanged** and remains the place to read prompt processing — cold
prefix cache, synthesised prompts at increasing depth, 1% low / median / 99% high. Nothing
changes in `results.json`: per-run `pp_tps` is still recorded, so older results re-render
under the new layout and no history is lost.

## 0.4.0

### Run one phase at a time

`run` measures three phases — single-stream decode, the prefill depth sweep, the concurrency
sweep — and until now it was all of them or a `--no-*` flag per phase you did not want.
Naming a phase now selects it and only it:

```bash
betterbench run --endpoint ... --model ... --prefill        # prompt processing only
betterbench run --endpoint ... --model ... --decode         # batch = 1 only
betterbench run --endpoint ... --model ... --concurrency    # the sweep only
betterbench run --endpoint ... --model ... --decode --prefill   # both, no sweep
```

With no phase named, nothing changes: all three run, and `--no-prefill` / `--no-concurrency`
work as before. Asking for and switching off the same phase in one command is an error rather
than a silent winner.

- **`--prefill` needs no corpus.** The depth sweep synthesises its own prompts, so a
  prefill-only run no longer exits on a corpus it was never going to read, and skips the
  `/v1/models` context probe when no prefill sweep will use it.
- **A `run_*: false` in a `--config` file is honoured.** `run_concurrency` was overwritten by
  the CLI default on every run, so only `--no-concurrency` could switch the sweep off; the
  config key had no effect. `run_single_stream` joins it as a config field.
- **The report says which phases a result holds.** A partial run's header carries a `phases`
  line, and `passes/cat` — a single-stream number — is dropped from a result with no
  single-stream section instead of describing a phase that never ran.

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
