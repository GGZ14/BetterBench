# BetterBench

A real-world, percentile-based benchmark for LLM inference servers. Point it at any
**OpenAI-compatible `/v1` endpoint** (vLLM, llama.cpp server, SGLang, TGI, Ollama, LM Studio, …)
and it measures how the stack *actually feels* across representative task categories —
prose, reasoning, math, code, JSON, file-edits, summarization, chat — not synthetic filler.

Unlike most tools that print one average tokens/sec at batch=1, BetterBench reports the
**distribution** (1% low / median / average / 99% high) for three layers — time-to-first-token,
inter-token latency, and per-run throughput — and is engineered to **resolve ~1% differences**
through paired, drift-cancelled A/B measurement.

Every run also writes a **self-contained HTML report** beside its `results.json` — headline
tiles, charts and the full tables in one file with no network assets, so it opens offline and
travels in an email.

<p align="center">
  <img src="docs/img/report-overview.png" width="900"
       alt="BetterBench HTML report: run header with the endpoint, corpus version and custom
            run notes; headline tiles for combined decode, update p99, TTFT p50, aggregate
            throughput and prefill; and a bar chart of decode throughput per category against
            the weighted combined score.">
</p>

**[→ Open the example report](https://htmlpreview.github.io/?https://github.com/GGZ14/BetterBench/blob/main/docs/example-report.html)**
&nbsp;·&nbsp; more screenshots in [The HTML report](#the-html-report)

## Install

```bash
pip install -e .          # editable install works out of the box; needs Python 3.10+, stdlib + numpy only
```

## Quickstart

```bash
# Benchmark one endpoint (single-stream, prefill and concurrency sweeps) and print
# the report
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 \
                --out results/radiance-27b.json

# Re-render a saved result later (markdown to stdout, or a charted HTML page)
betterbench report results/radiance-27b.json
betterbench report results/radiance-27b.json --html

# Run just one phase — decode, prompt processing, or the concurrency sweep
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --prefill
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --decode --concurrency

# Choose how many measured passes per category (default 20), or take the 5-pass shortcut
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --passes 10
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --quick

# Record what the run actually was, so two results can be compared later
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 \
                --note image=v0.9.3 --note quant=mxfp4 --note kv=fp8 --note tp=2

# Rigorous paired A/B between two endpoints (e.g. two images on ports 8080/8081)
betterbench ab --endpoint-a http://host:8080/v1 --endpoint-b http://host:8081/v1 \
               --model Qwen3.8 --mde 1.0
```

## What you get

**Single-stream table** — per category: TTFT p50/p99, decode t/s median ± IQR, a weighted
combined score, and a latency pair that depends on what the server actually streams:

- **one token per update** — ITL as tokens/sec (1% low = slowest tokens / median / 99% high
  = fastest), exactly as before;
- **several tokens per update** (speculative decoding — MTP, EAGLE, Medusa, n-gram,
  DFlash2) — `update p50 / p99 (ms)` and `tok/update` instead. Tokens that arrive together
  in one network write have no time *between* them, so BetterBench reports the gap it
  measured rather than inventing a per-token number. p99 is the stutter.

**Reasoning / answer split** — on a thinking model, TTFT is time to the first *thinking*
token, a wait nobody experiences. Where the split is detectable, the report adds the
reasoning share and **TTFA** — time to the first *answer* token — always with the `k/N` runs
it was computed over. A run cut off before `</think>` is recorded as `unknown`, not as an
answer.

**Prompt-processing (prefill) sweep** — prefill throughput (prompt tokens ÷ TTFT) at
increasing input depth (2K → 64K), with a cold prefix cache and tiny decode, so you see how
prompt processing scales for long-context / RAG / big-file workloads. A `PP t/s` column also
appears per category in the single-stream table.

**Run length you control** — `--passes N` sets the measured passes per category (default 20)
and `--warmup N` the discarded ones; both override whatever a `--config` file says. `--quick`
is the shorthand for a smoke run — 5 passes after 1 warmup — for checking that an endpoint,
model name and corpus are wired up before you spend an hour on the real thing. Five passes
is too thin a sample to publish or to compare stacks with; the pass count is printed in the
banner and recorded in the report so a quick result can't be mistaken for a full one.

**Concurrency sweep** — aggregate throughput and per-request TTFT/decode percentiles at
increasing load, revealing the throughput/latency knee.

**Phases you can run one at a time** — `--decode` (single-stream, batch = 1), `--prefill`
(the prompt-processing depth sweep) and `--concurrency` (the load sweep). Naming a phase runs
only the phases named, so re-measuring prompt processing after a kernel change costs the
prefill sweep and nothing else; name several to run several. With none named all three run,
as before. `--prefill` reads no corpus — the sweep builds its own prompts by depth — and a
partial result records which phases it holds, so a prefill-only file can't be mistaken for a
full run.

**A charted HTML report** — every `run` also writes a standalone `.html` beside its
`results.json` (same basename), charting every phase and carrying the full tables underneath.
See [The HTML report](#the-html-report) below. Skip it with `--no-html`, or point it somewhere
else with `--html-out`.

**Paired A/B** — Δ with a 95% confidence interval and a verdict that **refuses to call a
winner inside the noise band**:

```
Decode throughput (B vs A)
- Δ = +1.34%  (95% CI [+0.91%, +1.77%])
- B is 1.34% faster — SIGNIFICANT
```

## The HTML report

`betterbench run` writes the report automatically; `betterbench report results.json --html`
re-renders one from a saved result at any time. It is a **single file** — the charts are
hand-drawn SVG, so there is no chart library, no CDN, no fonts to fetch and no build step.
Open it offline, attach it to an email, drop it in a PR.

**[→ Open the example report](https://htmlpreview.github.io/?https://github.com/GGZ14/BetterBench/blob/main/docs/example-report.html)**
&nbsp;·&nbsp; [`docs/example-report.html`](docs/example-report.html)

### Every number is hoverable

The charts are the summary; the detail is one hover away — here, the per-category IQR,
coefficient of variation and pass count behind a single decode bar. A high CV means that
category's passes disagreed, which is exactly what you need to know before believing a small
difference.

<p align="center">
  <img src="docs/img/report-tooltip.png" width="860"
       alt="The decode-throughput chart with a tooltip open on the file-edit bar, showing
            decode 246.7 t/s, an IQR of 35.0, a coefficient of variation of 10.8%, and 5
            passes.">
</p>

### The concurrency knee, and prefill by depth

<table>
<tr>
<td width="50%" valign="top">
  <img src="docs/img/report-concurrency.png"
       alt="Two charts: aggregate throughput rising with concurrency level 1 to 16, and
            grouped TTFT p50 / p99 bars at each level showing the queueing tail open up.">
</td>
<td width="50%" valign="top">
  <img src="docs/img/report-prefill.png"
       alt="Prompt-processing throughput by input depth from 2K to 250K tokens, plotted as a
            line with a shaded 1%-low to 99%-high band.">
</td>
</tr>
<tr>
<td valign="top">Aggregate throughput against load, and the TTFT p50/p99 pair underneath it —
where the line flattens and the p99 bar takes off is the knee.</td>
<td valign="top">Median prefill t/s at increasing input depth against a cold prefix cache, with
the 1%-low → 99%-high band shaded.</td>
</tr>
</table>

### The full tables are still there

Charts are for reading; the tables are for quoting. Every figure that went into them is under
a `Full numbers` fold — single-stream per category, the concurrency sweep, the reasoning /
answer split, and the prefill sweep — with the `†` sample-size marks carried through, so an
under-sampled percentile is labelled wherever it appears.

<p align="center">
  <img src="docs/img/report-tables.png" width="900"
       alt="The Full numbers section: single-stream per-category table with TTFT, prefill,
            update p50/p99, tokens per update, decode median, IQR and CV; the concurrency
            sweep; the reasoning/answer split with TTFA; and the prefill sweep by depth.">
</p>

### It follows the reader's theme

No toggle to find and no setting to remember — the report renders light or dark from the
reader's own `prefers-color-scheme`.

<p align="center">
  <img src="docs/img/report-overview-dark.png" width="900"
       alt="The same report header, headline tiles and decode chart rendered in dark mode.">
</p>

> The screenshots above are a `--quick` smoke run (5 passes per category), which is why the
> percentiles carry `†` marks — see **Sample-size honesty** below.

## Why it can see 1%

Raw run-to-run noise on real stacks is commonly 3–8% — enough to bury a 1% signal. BetterBench
controls for it (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)):

- **Paired, interleaved A/B** on the same warm box → the CI is on the *difference*, cancelling
  thermal/cache drift.
- **Greedy + fixed seed** (A/B default) → identical token counts, no sampling variance.
- **Nonce prefixes** → honest prefill (no accidental prefix-cache hits).
- **Power analysis / run-to-confidence** → runs until the Δ CI is tighter than your target MDE.
- **Sample-size honesty** → every percentile below `n · tail ≥ 5` is marked `†` and the
  shortfall is recorded in `results.json`. The token-rich gap series is the trustworthy tail;
  a per-run p99 at 20 passes is labelled for what it is.
- **Null-test gate** → the tool proves it can't manufacture a phantom win before you trust it.

## Validate the harness (no GPU needed)

```bash
python tools/self_test.py
# checks timing accuracy against a mock SSE server (known TTFT/ITL)
# and the null A/B test (same endpoint vs itself must report "within noise")

pip install -e ".[dev]" && python -m pytest -q
# gap math under batched streams, the sample-size gate, the A/B sign convention,
# the reasoning split, phase selection end-to-end, and re-rendering a v0.2.3 results file
```

## Corpus

Versioned prompt sets live in [`corpus/v1/`](corpus/v1) (one JSONL per category), spanning an
input×output length grid so prefill-heavy (file-edit, summarize) and decode-heavy (story,
essay) shapes are both represented. Results are only comparable within a `corpus_version`.
See [`docs/CONTRIBUTING_PROMPTS.md`](docs/CONTRIBUTING_PROMPTS.md) to add prompts.

## License

Apache-2.0.
