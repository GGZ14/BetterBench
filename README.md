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
betterbench run --endpoint http://192.168.12.47:8080/v1 --model Qwen3.6 \
                --out results/radiance-27b.json

# Re-render a saved result later
betterbench report results/radiance-27b.json

# Rigorous paired A/B between two endpoints (e.g. two images on ports 8080/8081)
betterbench ab --endpoint-a http://host:8080/v1 --endpoint-b http://host:8081/v1 \
               --model Qwen3.6 --mde 1.0
```

## What you get

**Single-stream table** — per category: TTFT p50/p99, ITL as tokens/sec
(1% low = slowest tokens / median / 99% high = fastest), decode t/s median ± IQR, and a
weighted combined score.

**Prompt-processing (prefill) sweep** — prefill throughput (prompt tokens ÷ TTFT) at
increasing input depth (2K → 64K), with a cold prefix cache and tiny decode, so you see how
prompt processing scales for long-context / RAG / big-file workloads. A `PP t/s` column also
appears per category in the single-stream table.

**Concurrency sweep** — aggregate throughput and per-request TTFT/decode percentiles at
increasing load, revealing the throughput/latency knee.

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
