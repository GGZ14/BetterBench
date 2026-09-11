# Changelog

## Unreleased

### The concurrency sweep stops at 8

The default levels are `[1, 2, 4, 8]`, no longer `[1, 2, 4, 8, 16]`. Level 16
cost a fifth of the sweep's wall time to answer a question few people ask of a
single box, and the knee it was there to find has usually shown itself by 8.
Put it back — or go further — with `"concurrency_levels": [1, 2, 4, 8, 16, 32]`
in a `--config` file. Results recorded at 16 still render; the report draws the
levels the run actually contains.

### The prefill sweep no longer sends the same text twice

The filler behind the nonce was one paragraph repeated to length, which a
prefix cache could not reuse — its block hashes chain from the start of the
prompt, and the nonce broke the chain — but a cache keyed on block *content*
could reuse almost completely. Measured on Qwen3's tokenizer, an 80k-character
prompt was 918 blocks of 16 tokens with **25 distinct values among them**: a
per-layer LRU over KV blocks hit on essentially every block, within a single
pass and across every pass of the sweep, and the sweep timed cache lookups
instead of prompt processing.

The body is now a fresh random ordering of the same words on every pass — 917
of 917 blocks distinct, nothing shared between passes. Shuffling rather than
inventing text keeps the word multiset, and with it the characters-per-token
ratio, exactly what it was: measured depths move by under 1%, so results
recorded before this change remain comparable. `unique_nonce: false` still asks
for the old byte-identical prompt, for measuring a warm cache deliberately —
which the prefill sweep previously ignored.

## 0.5.0

**Upgrading:** two defaults moved. `run` without `--out` no longer writes
`results/run.json` into the directory you ran it from — it writes into a fresh
`~/.betterbench/runs/<timestamp>-<label>/` instead — and `ab` now saves a file
where it used to only print. A script that reads `results/run.json` from the
working directory should pass `--out results/run.json` explicitly; `--out` has
always won and still does.

### Runs live in `~/.betterbench/runs/`, each in its own versioned directory

`run` (and `ab`, which now saves by default) no longer writes `results/run.json`
into the directory you ran it in. Without `--out`, each run gets its own
directory under `~/.betterbench/runs/` — `$BETTERBENCH_HOME/runs/` when that
variable is set — named `<YYYYMMDD-HHMMSS>-<label>` where the timestamp is when
the run started and the label is the model name (`--name mxfp4-flip` to call it
something else): `results.json` plus the HTML report together, `ab.json` for A/B.
A second run in the same second appends `-2`, so re-running can never overwrite
an earlier result, and `--out PATH` always still wins for scripts. The directory
is allocated once the run clears validation and before it starts measuring, and
its path is printed up front: a run that has nowhere to write fails in its first
second rather than after hours of measurement it can no longer save. `report`/`compare` no longer create a stray
`results/x/` in the working directory.

### Authentication: `--api-key` / `--api-key-a` / `--api-key-b`

Requests went out unauthorised, which is right for a local vLLM/llama.cpp
server but wrong behind a gateway. A key sent on the wire as
`Authorization: Bearer KEY` now authenticates every request — including the
`/v1/models` context probe, which previously would have 401'd and read as
"unknown max context" — and is **never recorded** in the results, so the
JSON stays shareable. `export BETTERBENCH_API_KEY=...` fills in for any
missing key flag, keeping it out of shell history.

### A newer release is announced when there is one

BetterBench now checks once a day whether a newer version has been tagged and
says so after the run, because a benchmark a version behind can be measuring
something the current release already fixed. The check is on a background
thread that is joined **before the first measured request** — it can never be
on the wire while a request to the endpoint under test is being timed — its
answer is cached in `$BETTERBENCH_HOME/update-check.json` for 24 hours, and
every failure is silence rather than an error: it may never be the reason a
benchmark did not run. `--no-update-check` or `BETTERBENCH_NO_UPDATE_CHECK=1`
switches it off.

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
