# BetterBench

A real-world, percentile-based benchmark for LLM inference servers. Point it at any
**OpenAI-compatible `/v1` endpoint** (vLLM, llama.cpp server, SGLang, TGI, Ollama, LM Studio, …)
and it measures how the stack *actually feels* across representative task categories —
prose, reasoning, math, code, JSON, file-edits, summarization, chat — not synthetic filler.

Unlike most tools that print one average tokens/sec at batch=1, BetterBench reports the
**distribution** (1% low / median / average / 99% high) for three layers — time-to-first-token,
inter-token latency, and per-run throughput — and is engineered to **resolve ~1% differences**
through paired, drift-cancelled A/B measurement.

## Install

```bash
pip install -e .          # editable install works out of the box; needs Python 3.10+, stdlib + numpy only
```

## Quickstart

```bash
# Benchmark one endpoint (single-stream + concurrency sweep) and print the report
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 \
                --out results/radiance-27b.json

# Re-render a saved result later (markdown to stdout, or a charted HTML page)
betterbench report results/radiance-27b.json
betterbench report results/radiance-27b.json --html

# Choose how many measured passes per category (default 20), or take the 5-pass shortcut
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --passes 10
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.8 --quick

# Rigorous paired A/B between two endpoints (e.g. two images on ports 8080/8081)
betterbench ab --endpoint-a http://host:8080/v1 --endpoint-b http://host:8081/v1 \
               --model Qwen3.8 --mde 1.0
```

## What you get

**Single-stream table** — per category: TTFT p50/p99, ITL as tokens/sec
(1% low = slowest tokens / median / 99% high = fastest), decode t/s median ± IQR, and a
weighted combined score.

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

**A charted HTML report** — every `run` also writes a standalone `.html` beside its
`results.json` (same basename), so the numbers land as something you can actually read: headline
tiles, decode throughput per category against the combined score, the ITL 1%-low → 99%-high range
per category, the concurrency knee, TTFT p50 vs p99, and prefill throughput by depth — plus the
full tables. One self-contained file with no network assets, so it opens offline and travels in an
email; it needs no extra dependencies and follows the reader's light/dark theme. Skip it with
`--no-html`, or point it somewhere else with `--html-out`.

**Paired A/B** — Δ with a 95% confidence interval and a verdict that **refuses to call a
winner inside the noise band**:

```
Decode throughput (B vs A)
- Δ = +1.34%  (95% CI [+0.91%, +1.77%])
- B is 1.34% faster — SIGNIFICANT
```

## Why it can see 1%

Raw run-to-run noise on real stacks is commonly 3–8% — enough to bury a 1% signal. BetterBench
controls for it (see [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md)):

- **Paired, interleaved A/B** on the same warm box → the CI is on the *difference*, cancelling
  thermal/cache drift.
- **Greedy + fixed seed** (A/B default) → identical token counts, no sampling variance.
- **Nonce prefixes** → honest prefill (no accidental prefix-cache hits).
- **Power analysis / run-to-confidence** → runs until the Δ CI is tighter than your target MDE.
- **Sample-size honesty** → ITL (token-rich) is the trustworthy tail metric; per-run p99 is only
  shown when the run count supports it.
- **Null-test gate** → the tool proves it can't manufacture a phantom win before you trust it.

## Validate the harness (no GPU needed)

```bash
python tools/self_test.py
# checks timing accuracy against a mock SSE server (known TTFT/ITL)
# and the null A/B test (same endpoint vs itself must report "within noise")
```

## Corpus

Versioned prompt sets live in [`corpus/v1/`](corpus/v1) (one JSONL per category), spanning an
input×output length grid so prefill-heavy (file-edit, summarize) and decode-heavy (story,
essay) shapes are both represented. Results are only comparable within a `corpus_version`.
See [`docs/CONTRIBUTING_PROMPTS.md`](docs/CONTRIBUTING_PROMPTS.md) to add prompts.

## License

Apache-2.0.
