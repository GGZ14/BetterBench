# BetterBench — a real-world LLM inference benchmark suite

A plan for an open-source, percentile-based inference benchmark that measures how a
serving stack *actually feels* across representative task categories, against any
OpenAI-compatible `/v1` endpoint.

**Status:** implemented (v0.4.0) · **License:** Apache-2.0 · **Language:** Python 3.10+

---

## 1. Goals and non-goals

**Goals**

- Measure inference *serving* performance with prompts that resemble real work — prose,
  reasoning, math, code, JSON, file-edits — not synthetic fixed-length filler.
- Report the distribution, not just the mean: TTFT, inter-token latency (ITL), and
  end-to-end tokens/sec, summarized as **1% low / median / average / 99% high** per
  category and combined.
- Exercise both **single-stream** (how one request feels) and a **concurrency sweep**
  (how the server behaves under multi-user load).
- Target **any OpenAI-compatible `/v1` endpoint** — vLLM, llama.cpp server, SGLang, TGI,
  Ollama, LM Studio — so any card/engine can be compared apples-to-apples.
- Be reproducible and shareable: versioned prompt corpus, machine-readable results,
  full environment capture, and a results format others can submit against.
- **Resolve small wins.** Be repeatable enough to measure a **~1% true improvement** with
  statistical confidence — through paired A/B measurement, variance control, and power
  analysis (see §7). A benchmark that can't tell 1% from noise can't guide tuning.

**Non-goals (v1)**

- Output *quality/correctness* scoring. This suite is **speed-only** by design; a fast-but-wrong
  config is out of scope for v1 (a future `--verify` mode is noted in the roadmap).
- Training/fine-tuning benchmarks.
- Engine-internal profiling (kernel timings) — that's what rocprof/nsys are for; BetterBench
  measures the black-box endpoint.

---

## 2. What it measures (metric definitions)

All timing is captured **client-side** from a streaming response, cross-checked against the
server's reported `usage`. Three layers:

### 2.1 Time to first token (TTFT) — the prefill/queue experience
- `TTFT = t(first_token_chunk) − t(request_sent)`.
- Dominated by prompt length (prefill) at low load, and by queueing at high concurrency.
- Reported per category and per input-length bucket.

### 2.2 Stream-update gap and inter-token latency — decode smoothness
- For each streamed update after the first, record its arrival time. The **gap between
  updates is the measurement**; nothing is derived from it by division.
- Classify each record as one-token-per-update or batched (§ chunk↔token in METHODOLOGY).
  - **One token per update.** The update gap *is* the inter-token latency. Report it as
    today: **median tok/s** = `1/p50(ITL)`, **"99% high" tok/s** = `1/p1(ITL)`, and a
    gaming-style **"1% low"** = mean of the worst 1% of per-token rates (`0.1%-low` is
    computed but not printed). Note these are two different estimators; the column headers
    say which is which.
  - **Several tokens per update.** Report **update p50 / p99 (ms)** and **tok/update**, and
    emit no per-token latency at all.
- This supersedes the original design, which specified
  `ITL_per_token = Δt_chunk / tokens_in_chunk`. That formula is the origin of the defect
  fixed in 0.3.0 — and note the implementation was worse still, applying a single per-run
  scalar. Even the per-chunk version invents arrival times for tokens that arrived together
  in one network write. A tail statistic computed from invented arrival times is not a
  measurement of anything.
- Because a single long generation yields hundreds–thousands of gap samples, category-level
  tail percentiles are far better supported than per-run ones — but batching by 4x means 4x
  fewer samples, so the gate in §6.3 applies here too and thin categories are marked.

### 2.3 End-to-end tokens/sec (per run) — the throughput number
- `decode_tps = completion_tokens / (t_last_token − t_first_token)`.
- `total_tps  = completion_tokens / (t_last_token − t_request_sent)` (includes TTFT).
- Percentiles here (p1/p50/p99 across runs) need **many runs** to be meaningful (see §6.3),
  so per-run t/s is reported primarily as **median + IQR**, with p1/p99 shown only when the
  run count supports it.

### 2.4 Supporting counters
- Prompt tokens, completion tokens (including reasoning tokens — see §6.1), finish reason,
  achieved concurrency, and (at load) aggregate server throughput = Σ completion_tokens / wall.

---

## 3. Prompt corpus design (the heart of "real-world")

The corpus is what makes this representative. Design principles:

### 3.1 Categories (versioned JSONL, one file each)
- **prose / literary** — creative writing, continuation in a given voice, editing passages.
- **reasoning** — multi-step logic, planning, common-sense, "explain then answer."
- **math** — arithmetic, word problems, light proofs, across difficulty tiers.
- **code** — generation, completion, and debugging across a few languages (Py/JS/Rust/SQL).
- **json** — schema-constrained structured output and function-call-style payloads.
- **file-edit** — agentic diff/patch and search-replace over a supplied file (big prefill,
  small decode — the classic coding-agent shape).
- **summarization** — long input → short output (prefill-heavy, TTFT-dominated).
- **chat / multi-turn** — a short conversation history then a reply (realistic context reuse).

### 3.2 The input × output length matrix (critical)
Real work spans wildly different prefill:decode ratios, and TTFT vs ITL matter differently
for each. Every category carries prompts spanning a 2-D grid:

| | short output (~64) | medium (~512) | long (~2k+) |
|---|---|---|---|
| **short input (<512)** | chat turn | prose burst | story writing |
| **medium input (2–8k)** | json extract | code gen | reasoning essay |
| **long input (16–64k)** | needle/lookup | summarize | large refactor |

This is what separates BetterBench from "one 128-token prompt": a file-edit at 32k input / 200
output stresses prefill + TTFT, while story-writing at 200 input / 3k output stresses decode
ITL. The report breaks results down by this grid so a stack's strengths/weaknesses are visible.

### 3.3 Anti-skew rules (learned the hard way)
- **Prefix-cache control.** Repeated identical prompts hit the server's prefix cache and
  report unrealistically fast prefill. Each repetition injects a unique nonce prefix
  (`[run a3f9c1] …`) by default so prefill is measured honestly. A `--warm-cache` mode can
  additionally report the cached-prefill case, labeled separately — never mixed.
- **Output-length normalization.** Under real sampling, output length varies run to run, which
  skews per-request numbers. BetterBench reports **per-token** rates (immune to length) as
  primary, always alongside the token counts, and caps `max_tokens` per grid cell so runs are
  comparable.
- **Sampling.** Default realistic sampling (`temp 0.7, top_p 0.95, top_k 20`) with a fixed
  per-run seed where the server honors it; a `--greedy` mode (temp 0) is available for
  tight reproducibility. Sampling params are recorded in results and pinned per corpus version.

### 3.4 Corpus governance
- Corpus is **versioned** (`corpus_version: "1.0"`); results are only comparable within a
  version. Prompts live in JSONL with `id`, `category`, `input_len_bucket`,
  `output_len_bucket`, `messages`, `max_tokens`, and optional `attachments` (for file-edit).
- A `CONTRIBUTING_PROMPTS.md` defines how the community proposes additions (representative,
  license-clean, no PII, deterministic length target).

---

## 4. Load profiles

1. **Single-stream (batch = 1).** The clean latency picture: TTFT + ITL percentiles + per-run
   t/s, per category and grid cell. This is the headline table.
2. **Concurrency sweep.** Fire N concurrent requests (N ∈ {2, 4, 8, 16, 32}, configurable),
   drawing from the corpus. Report, per level: aggregate server tok/s, per-request TTFT
   percentiles (queueing shows up here), per-request ITL percentiles (contention shows up
   here), and achieved vs requested concurrency. This reveals the throughput/latency knee.

Each level runs a fixed number of **completed** requests per category (default 20 single-stream
warm samples after 3 warmups; concurrency levels run a fixed duration or request budget).

---

## 5. Harness architecture

```
realbench/
  client.py     # async httpx client; SSE streaming; per-chunk timestamps; usage capture
  runner.py     # orchestrates categories × grid × load-profiles × repetitions; warmup
  metrics.py    # TTFT / ITL / tps; percentiles; 1%-low & 0.1%-low; per-cat + combined
  corpus/*.jsonl
  config/default.yaml
  report.py     # results.json → markdown + HTML + charts
  schema.py     # versioned results schema
  env.py        # capture: engine+version, model, quant, GPU, driver, flags, host
```

- **Async streaming client.** Uses `stream: true` and `stream_options: {include_usage: true}`
  so the final chunk carries authoritative token counts. Records a monotonic timestamp per
  chunk. Counts **both** `delta.content` and reasoning-channel deltas (see §6.1).
- **Runner.** Warms up (discarded), then runs to completion the configured repetitions;
  handles the concurrency sweep via a bounded async pool; supports request budget or wall-clock
  budget per level; resumable.
- **Determinism of measurement.** Recommends running the client on the same host or a
  low-latency link; records both client timing and server `usage` so network jitter is
  detectable. Optional `--server-timing` reads any `x-*-latency` headers the engine exposes.

**CLI**
```
realbench run   --endpoint http://192.168.12.47:8080/v1 --model Qwen3.6 \
                --config config/default.yaml --out results/radiance-27b.json
realbench report results/radiance-27b.json --format md,html
realbench compare results/*.json          # side-by-side table across stacks
```

---

## 6. Statistics and rigor

### 6.1 Reasoning tokens
Models like Qwen3.6 emit a separate reasoning channel (`reasoning` / `reasoning_content`).
The harness **counts reasoning tokens as generated tokens** for ITL and t/s (they are real
decode work), TTFT is time to the first token of *any* channel, and the report shows the
reasoning vs answer token split per category alongside **TTFA** — time to the first *answer*
token, which is the wait a reader actually feels (thinking can dominate short prompts).

The split is detected from the channel switch or from an inline `</think>`, and apportioned
from the server's single token total by character count, so it is an estimate. A run
truncated before any answer began reports `reasoning_source="unknown"` with the split left
as `None` — never zeros, which would credit an answer that never arrived. Since most runs
under a tight `max_tokens` end that way, the split table always shows the `k/N` runs it is
computed over and withholds a figure it cannot support.

### 6.2 Warmup and cache state
- Discard the first K runs per category (cold compile / autotune / graph capture).
- Prefix-cache defaults to *cold* (nonce prefixes); warm is a separate, labeled run.
- Record whether the endpoint reports a prefix-cache hit rate; surface it in results.

### 6.3 Sample-size honesty (a differentiator)
Tail percentiles need enough samples or they're noise:
- **ITL percentiles** — token-rich; a few long generations give thousands of samples →
  1%-low is trustworthy from ~20 runs/category. This is why ITL is the primary tail metric.
- **Per-run t/s percentiles** — one sample per run; a real p99 needs 500 samples under the
  `n · tail ≥ 5` rule. BetterBench **marks** every percentile below that threshold with a `†`
  and records the shortfall under `sample_gate` in `results.json`. It marks rather than
  suppresses: TTFT p99 fails the rule at every pass count the tool offers, and deleting the
  column would cost more than labelling it honestly does.

### 6.4 Aggregation
- Per **category**, per **grid cell**, and **combined** (weighted by a declared real-world
  mix, e.g. code 30 / reasoning 20 / prose 15 / json 15 / file-edit 10 / summarize 10 —
  configurable, disclosed in the report).
- Report central tendency (median, mean) and spread (IQR, p1, p99, 1%-low, 0.1%-low, stdev).

### 6.5 Reproducibility metadata (captured automatically)
Engine name+version, model id+quant, KV-cache dtype, TP size, GPU model+count, driver/ROCm/CUDA
version, kernel, key server flags if discoverable, corpus version, suite git commit, sampling
params, timestamp, and a content hash of the exact prompts used. Embedded in `results.json`.

---

## 7. Repeatability: resolving ~1% differences (a core design driver)

To call a ~1% improvement *real*, the uncertainty on the measured **difference** must be well
under 1%. On these stacks, raw run-to-run noise is commonly **3–8%** (cache warmth, GPU
clock/thermal drift, scheduler nondeterminism, sampling-induced length variance) — that alone
buries a 1% signal. BetterBench gets there not by brute precision but by controlling variance and
comparing in pairs.

**A. Paired, interleaved A/B — the primary lever.** Never compare two configs from separate
sessions; session-to-session drift dominates. `compare --ab` runs config A and config B
**interleaved on the same warmed box** (A,B,A,B,… or randomized blocks) and computes the
*paired per-trial difference*. Common-mode drift cancels, so the confidence interval is on Δ,
not on A and B independently — routinely turning an unresolvable ±5% into a ±0.3% CI on the
difference. (This is precisely the discipline missing when you benchmark A today and B tomorrow.)

**B. Kill sampling variance.** The repeatability mode defaults to **greedy (temp 0) + fixed
seed + identical prompts**, so completion token counts are identical run-to-run and
length-induced variance vanishes. Sampled mode stays for realism, but 1%-resolution claims
require greedy.

**C. Thermal / clock steady-state.** GPU boost clocks track temperature and power; a cold card
runs fast then throttles. BetterBench warms to a **thermal plateau** before measuring and records
clock/temp/power per run (endpoint host or sidecar). Optionally pin clocks (disable DVFS / fix
SCLK) for maximum stability; whether pinned is recorded. Runs that drift beyond tolerance are
flagged and excluded.

**D. Quiet, pinned environment.** Idle box, CPU governor at performance, a single measurement
client, localhost/dedicated link. Background load is recorded; the full fingerprint is captured
per §6.5.

**E. Power analysis / auto-run-to-confidence.** From the observed coefficient of variation,
BetterBench computes the runs needed for a target **minimum detectable effect (MDE)** — default
**1% at 95% confidence** — as `n ≈ (z·CV / MDE)²`, using the (much smaller) CV of the paired
*difference*. It can **sequentially run until the Δ CI is tighter than the MDE**, then stop, so
you never guess the run count.

**F. Report significance, never vibes.** Every comparison prints the effect with a
bootstrap/t confidence interval and a verdict, e.g. `decode: +1.3% ± 0.4% (95% CI) —
SIGNIFICANT` vs `+0.6% ± 1.4% — within noise, not distinguishable`. BetterBench refuses to declare
a winner when the CI straddles zero.

**G. Lead with the tightest metric.** ITL **median** (thousands of token samples) is far more
stable than per-run t/s (one sample per run), so the repeatability mode leads with paired
ITL-median deltas and reports per-run t/s deltas secondarily.

**H. Null-test gate (validate the harness itself).** Run the *same* config as both A and B; a
correct suite must report the difference as **not significant (~0% ± <1%)**. If it flags a
phantom win, the noise model is wrong. This null test ships in CI so the tool's own
resolution claims stay honest.

---

## 8. Output and reporting

- **`results.json`** — machine-readable, schema-versioned, the source of truth (safe to submit
  to a shared leaderboard).
- **Markdown report** — the headline tables:
  - single-stream per category: TTFT p50/p99, ITL 1%-low/median/99%-high (as tok/s),
    decode t/s median±IQR;
  - concurrency sweep: the throughput-vs-latency knee;
  - combined weighted summary.
- **HTML report** — same data plus charts (ITL distribution violin/CDF per category;
  throughput-vs-concurrency curve; TTFT-vs-input-length). Charts via a small vendored JS lib,
  no server needed.
- **`compare`** — diff two or more result files, or run a live paired A/B (`--ab`, §7). Always
  reports each delta with a confidence interval and a significance verdict, and never declares a
  winner inside the noise band (e.g. `+1.3% ± 0.4% — SIGNIFICANT` / `+0.6% ± 1.4% — within noise`).

---

## 9. Repo, packaging, community

- `pyproject.toml`, installable via `uv`/`pip`; zero heavyweight deps (httpx, numpy; the HTML report is hand-rolled, no template engine).
- `README.md` with a 30-second quickstart and an example result table.
- `docs/METHODOLOGY.md` (metric definitions + the sample-size rules), `docs/RESULTS_FORMAT.md`,
  `docs/CONTRIBUTING_PROMPTS.md`.
- **Reference dataset:** ship a first results set generated on the radiance R9700 stack
  (27B + 35B) so the community has a real baseline to compare against — good dogfooding and a
  genuine contribution.
- **Leaderboard (later):** a `results/` submission format + a simple aggregator; comparability
  guarded by corpus version + disclosed config.

---

## 10. Phased roadmap

- **P0 — Spec & scaffold.** Finalize metric defs; repo skeleton; config schema; CI lint.
- **P1 — Core harness (single-stream).** Streaming client with accurate per-chunk timing;
  TTFT/ITL/tps + percentiles; one category end-to-end. *Exit test:* timing validated against a
  mock server with known delays (±2% error).
- **P2 — Full corpus.** All categories × input/output grid; nonce/cache control; reasoning-token
  counting; greedy + sampled modes.
- **P3 — Concurrency sweep.** Bounded async pool; per-level aggregation; knee detection.
- **P4 — Reporting.** Markdown + HTML + charts; `compare`; reproducibility metadata capture.
- **P5 — Rigor & repeatability pass (§7).** Paired A/B mode, greedy determinism, thermal
  steady-state + clock/temp capture, power analysis / auto-run-to-confidence, significance
  reporting, and the null-test CI gate. *Exit test:* null test (same config as A and B) reports
  ~0% ± <1%, and a known real change is flagged SIGNIFICANT.
- **P6 — Release v0.1.** Reference results on radiance; CONTRIBUTING; Apache-2.0; tag.
- **P7 — Community.** Submission format, corpus v1.1 process, optional leaderboard.

---

## 11. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Client-side timing jitter | Localhost/low-latency link; warmup; many samples; cross-check server `usage`; report client vs server delta |
| Prefix-cache inflates prefill | Nonce prefixes by default; warm-cache is a separate labeled run |
| Output-length variance skews per-request numbers | Per-token metrics primary; `max_tokens` caps per grid cell; token counts always shown |
| Tail percentiles from too few samples | Mark with † + record the shortfall in `sample_gate`; gap series (token-rich) is the primary tail metric |
| Reasoning tokens miscounted | Count all channels; report reasoning/answer split and TTFA; report `unknown` rather than zero when truncated mid-thought |
| Cross-engine unfairness | Pin corpus version; disclose full config; standardized warmup; identical sampling |
| Chunk≠token (servers batch tokens per SSE chunk) | Detect it and report a different metric — update p50/p99 + tok/update. Never normalize a gap into a per-token latency: the tokens in an update arrived together |
| Thermal/clock drift buries a 1% signal | Thermal-plateau warmup; per-run clock/temp/power capture; optional clock pinning; drift-tolerance exclusion (§7C) |
| A-vs-B session drift falsely credits/penalizes a change | Interleaved paired A/B with CI on the difference; null-test gate; never compare across sessions (§7A, §7H) |
| Under-powered comparison claims a phantom win | Power analysis sets required n for a 1% MDE; auto-run-to-confidence; significance verdicts refuse noise-band winners (§7E, §7F) |

---

## 12. Open questions to settle in P0
- Default real-world category weights for the combined score (proposed in §6.4 — confirm).
- Concurrency levels and per-level budget (request count vs wall-clock).
- Whether to ship a tiny "smoke" corpus (fast CI) plus the full corpus.
- Naming + GitHub org for the repo.
